#!/usr/bin/env node
/**
 * moray-surgeon TypeScript bridge.
 *
 * One JSON-RPC 2.0 request arrives on stdin as a single UTF-8 line; exactly one
 * JSON-RPC response leaves on stdout as a single UTF-8 line. Diagnostics and
 * crashes go to stderr, never to stdout, so the parent can parse stdout
 * strictly. The process exits 0 whenever it produced a response (including a
 * JSON-RPC error); a non-zero exit means the bridge itself failed to start
 * (for example because ts-morph is not installed).
 *
 * Methods and results:
 *
 *   ping              -> { pong: true, node: string, typeScript: string }
 *   parse             -> { statements: number, diagnostics: Diagnostic[] }
 *   verify            -> { ok: boolean, diagnostics: Diagnostic[] }
 *   rename            -> { changes: number, files: [{ path, text }] }
 *   extract_function  -> { source: string, changes: number }
 *   extract_variable  -> { source: string, changes: number }
 *   replace_call      -> { source: string, changes: number }
 *   add_import        -> { source: string, changes: number }
 *   remove_import     -> { source: string, changes: number }
 *
 * Diagnostic: { code: number, message: string, line: number, column: number }
 *   line is 1-based, column is 0-based. Columns count UTF-16 code units, which
 *   is what JavaScript string indices and the TypeScript compiler use.
 *
 * Errors:
 *   -32602 invalid params        (the parent turns this into ValueError)
 *   -32000 operation failed      (no definition, empty range, ... -> ValueError)
 *   -32603 internal bridge error (the parent raises TypeScriptBridgeError)
 *
 * Paths in a rename result are repository-relative POSIX paths. This process
 * never writes a file: the Python side validates every path again and performs
 * the atomic write, so the bridge cannot escape the repository.
 */

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { Node, Project, SyntaxKind, ts } from "ts-morph";

const INVALID_PARAMS = -32602;
const APPLICATION_ERROR = -32000;
const INTERNAL_ERROR = -32603;
const MAX_STACK_CHARS = 4000;

class BridgeError extends Error {
  constructor(code, message) {
    super(message);
    this.name = "BridgeError";
    this.code = code;
  }
}

function fail(code, message) {
  throw new BridgeError(code, message);
}

function requireObject(params) {
  if (params === null || typeof params !== "object" || Array.isArray(params)) {
    fail(INVALID_PARAMS, "params must be an object");
  }
  return params;
}

function requireString(params, key) {
  const value = params[key];
  if (typeof value !== "string" || value.length === 0) {
    fail(INVALID_PARAMS, `'${key}' must be a non-empty string`);
  }
  return value;
}

function optionalString(params, key) {
  const value = params[key];
  if (value === undefined || value === null) return null;
  if (typeof value !== "string" || value.length === 0) {
    fail(INVALID_PARAMS, `'${key}' must be a non-empty string when present`);
  }
  return value;
}

function requireInteger(params, key) {
  const value = params[key];
  if (!Number.isInteger(value)) {
    fail(INVALID_PARAMS, `'${key}' must be an integer`);
  }
  return value;
}

const IDENTIFIER = /^[A-Za-z_$][A-Za-z0-9_$]*$/;
const RESERVED = new Set([
  "break", "case", "catch", "class", "const", "continue", "debugger", "default",
  "delete", "do", "else", "enum", "export", "extends", "false", "finally", "for",
  "function", "if", "import", "in", "instanceof", "new", "null", "return", "super",
  "switch", "this", "throw", "true", "try", "typeof", "var", "void", "while", "with",
  "as", "implements", "interface", "let", "package", "private", "protected", "public",
  "static", "yield", "await", "async", "type", "namespace", "declare", "abstract",
]);

function requireIdentifier(params, key) {
  const value = requireString(params, key);
  if (!IDENTIFIER.test(value) || RESERVED.has(value)) {
    fail(INVALID_PARAMS, `'${key}' must be a valid TypeScript identifier`);
  }
  return value;
}

function syntaxDiagnostics(source, fileName) {
  const result = ts.transpileModule(source, {
    fileName,
    reportDiagnostics: true,
    compilerOptions: {
      target: ts.ScriptTarget.ESNext,
      module: ts.ModuleKind.ESNext,
    },
  });
  const errors = (result.diagnostics ?? []).filter(
    (diagnostic) => diagnostic.category === ts.DiagnosticCategory.Error,
  );
  if (errors.length === 0) return [];
  const sourceFile = ts.createSourceFile(
    fileName,
    source,
    ts.ScriptTarget.Latest,
    true,
  );
  return errors.map((diagnostic) => {
    const position = diagnostic.start ?? 0;
    const { line, character } = ts.getLineAndCharacterOfPosition(
      sourceFile,
      position,
    );
    return {
      code: diagnostic.code,
      message: ts.flattenDiagnosticMessageText(diagnostic.messageText, "\n"),
      line: line + 1,
      column: character,
    };
  });
}

function parseSource(source, fileName) {
  const sourceFile = ts.createSourceFile(
    fileName,
    source,
    ts.ScriptTarget.Latest,
    true,
  );
  return sourceFile;
}

function inMemoryProject(source) {
  const project = new Project({ useInMemoryFileSystem: true });
  return project.createSourceFile("input.ts", source);
}

function applyTextEdits(source, edits) {
  let output = source;
  const ordered = [...edits].sort(
    (left, right) => right.start - left.start || right.end - left.end,
  );
  for (const edit of ordered) {
    output = output.slice(0, edit.start) + edit.text + output.slice(edit.end);
  }
  return output;
}

function toPosix(value) {
  return value.split(path.sep).join("/");
}

function safeJoin(repoRoot, relativePath) {
  const absolute = path.resolve(repoRoot, relativePath);
  const relative = path.relative(repoRoot, absolute);
  if (relative === "" || relative.startsWith("..") || path.isAbsolute(relative)) {
    fail(INVALID_PARAMS, `path escapes the repository: ${relativePath}`);
  }
  return absolute;
}

function isEditable(filePath, repoRoot) {
  const relative = path.relative(repoRoot, filePath);
  if (relative === "" || relative.startsWith("..") || path.isAbsolute(relative)) {
    return false;
  }
  const segments = relative.split(path.sep).map((part) => part.toLowerCase());
  return !segments.includes("node_modules") && !segments.includes(".git");
}

function loadProject(repoRoot) {
  const tsconfig = path.join(repoRoot, "tsconfig.json");
  if (fs.existsSync(tsconfig)) {
    return new Project({ tsConfigFilePath: tsconfig });
  }
  const project = new Project({
    compilerOptions: {
      target: ts.ScriptTarget.ESNext,
      module: ts.ModuleKind.ESNext,
      noEmit: true,
      skipLibCheck: true,
    },
  });
  project.addSourceFilesAtPaths([
    "**/*.ts",
    "**/*.tsx",
    "**/*.mts",
    "**/*.cts",
    "!**/node_modules/**",
    "!**/dist/**",
    "!**/build/**",
    "!**/.git/**",
    "!**/.moray-cache/**",
  ]);
  return project;
}

function isDefinitionName(identifier) {
  let declarations = [];
  try {
    declarations = identifier.getSymbol()?.getDeclarations() ?? [];
  } catch {
    return false;
  }
  return declarations.some((declaration) => {
    if (
      Node.isImportSpecifier(declaration) ||
      Node.isImportClause(declaration) ||
      Node.isNamespaceImport(declaration)
    ) {
      return false;
    }
    if (
      declaration.getSourceFile().getFilePath() !==
      identifier.getSourceFile().getFilePath()
    ) {
      return false;
    }
    return declaration.getNameNode?.() === identifier;
  });
}

function offsetOf(sourceFile, line, column, label) {
  const lineStarts = sourceFile.compilerNode.getLineStarts();
  if (line < 1 || line > lineStarts.length) {
    fail(INVALID_PARAMS, `${label} line ${line} is outside the file`);
  }
  const lineStart = lineStarts[line - 1];
  const lineEnd =
    line < lineStarts.length
      ? lineStarts[line]
      : sourceFile.getFullText().length;
  const offset = lineStart + column;
  if (column < 0 || offset > lineEnd) {
    fail(INVALID_PARAMS, `${label} column ${column} is past the end of line ${line}`);
  }
  return offset;
}

const handlers = {
  ping() {
    return {
      pong: true,
      node: process.version,
      typeScript: ts.version,
    };
  },

  parse(params) {
    const source = requireString(params, "source");
    const fileName = optionalString(params, "fileName") ?? "input.ts";
    const sourceFile = parseSource(source, fileName);
    return {
      statements: sourceFile.statements.length,
      diagnostics: syntaxDiagnostics(source, fileName),
    };
  },

  verify(params) {
    const source = requireString(params, "source");
    const fileName = optionalString(params, "fileName") ?? "input.ts";
    const diagnostics = syntaxDiagnostics(source, fileName);
    return { ok: diagnostics.length === 0, diagnostics };
  },

  rename(params) {
    const repoRoot = path.resolve(requireString(params, "repoRoot"));
    const targetFile = requireString(params, "targetFile");
    const oldName = requireIdentifier(params, "oldName");
    const newName = requireIdentifier(params, "newName");
    if (oldName === newName) {
      fail(INVALID_PARAMS, "oldName and newName must differ");
    }
    const absoluteTarget = safeJoin(repoRoot, targetFile);
    if (!fs.existsSync(absoluteTarget)) {
      fail(APPLICATION_ERROR, `target file not found: ${targetFile}`);
    }
    const project = loadProject(repoRoot);
    if (!project.getSourceFile(absoluteTarget)) {
      project.addSourceFileAtPath(absoluteTarget);
    }
    const sourceFile = project.getSourceFileOrThrow(absoluteTarget);
    const definitions = sourceFile
      .getDescendantsOfKind(SyntaxKind.Identifier)
      .filter(
        (identifier) =>
          identifier.getText() === oldName && isDefinitionName(identifier),
      );
    if (definitions.length === 0) {
      fail(
        APPLICATION_ERROR,
        `no definition of '${oldName}' found in ${targetFile}`,
      );
    }

    const seen = new Set();
    const perFile = new Map();
    const languageService = project.getLanguageService();
    for (const definition of definitions) {
      for (const referenced of languageService.findReferences(definition)) {
        for (const reference of referenced.getReferences()) {
          const referenceFile = reference.getSourceFile();
          const filePath = referenceFile.getFilePath();
          if (!isEditable(filePath, repoRoot)) continue;
          const span = reference.getTextSpan();
          const key = `${filePath}:${span.getStart()}:${span.getLength()}`;
          if (seen.has(key)) continue;
          seen.add(key);
          let entry = perFile.get(filePath);
          if (entry === undefined) {
            entry = { sourceFile: referenceFile, spans: [] };
            perFile.set(filePath, entry);
          }
          entry.spans.push({ start: span.getStart(), end: span.getEnd() });
        }
      }
    }
    if (seen.size === 0) {
      fail(APPLICATION_ERROR, `no references found for '${oldName}'`);
    }

    const files = [];
    for (const entry of perFile.values()) {
      const text = applyTextEdits(
        entry.sourceFile.getFullText(),
        entry.spans.map((span) => ({ ...span, text: newName })),
      );
      files.push({
        path: toPosix(path.relative(repoRoot, entry.sourceFile.getFilePath())),
        text,
      });
    }
    if (files.length === 0) {
      fail(APPLICATION_ERROR, `'${oldName}' was not renamed in ${targetFile}`);
    }
    return { changes: seen.size, files };
  },

  extract_function(params) {
    const source = requireString(params, "source");
    const functionName = requireIdentifier(params, "functionName");
    const startLine = requireInteger(params, "startLine");
    const endLine = requireInteger(params, "endLine");
    if (startLine > endLine) {
      fail(INVALID_PARAMS, "startLine must be <= endLine");
    }
    const project = new Project({ useInMemoryFileSystem: true });
    const sourceFile = project.createSourceFile("input.ts", source);
    const statements = sourceFile
      .getStatements()
      .filter(
        (statement) =>
          statement.getStartLineNumber() >= startLine &&
          statement.getEndLineNumber() <= endLine,
      );
    if (statements.length === 0) {
      fail(
        APPLICATION_ERROR,
        `no statements fully inside lines ${startLine}-${endLine}`,
      );
    }
    const indent = project.manipulationSettings.getIndentationText();
    const body = statements
      .map((statement) => indent + statement.getText())
      .join("\n");
    const replacement =
      `function ${functionName}() {\n${body}\n}\n` +
      `${functionName}();`;
    const first = statements[0];
    const last = statements[statements.length - 1];
    return {
      source:
        source.slice(0, first.getStart()) +
        replacement +
        source.slice(last.getEnd()),
      changes: statements.length,
    };
  },

  extract_variable(params) {
    const source = requireString(params, "source");
    const variableName = requireIdentifier(params, "variableName");
    const startLine = requireInteger(params, "startLine");
    const startColumn = requireInteger(params, "startColumn");
    const endLine = requireInteger(params, "endLine");
    const endColumn = requireInteger(params, "endColumn");
    const project = new Project({ useInMemoryFileSystem: true });
    const sourceFile = project.createSourceFile("input.ts", source);
    const start = offsetOf(sourceFile, startLine, startColumn, "start");
    const end = offsetOf(sourceFile, endLine, endColumn, "end");
    if (start >= end) {
      fail(INVALID_PARAMS, "the range must not be empty");
    }
    const matches = sourceFile
      .getDescendants()
      .filter(
        (node) =>
          node.getStart() === start &&
          node.getEnd() === end &&
          Node.isExpression(node),
      );
    if (matches.length === 0) {
      fail(APPLICATION_ERROR, "no expression occupies the given range");
    }
    // Descendants are visited pre-order, so the first match is the outermost
    // node with this exact span (wrapping parentheses etc. come first).
    const expression = matches[0];
    const statement = expression.getFirstAncestor((node) =>
      Node.isStatement(node),
    );
    if (statement === undefined) {
      fail(APPLICATION_ERROR, "the expression is not inside a statement");
    }
    const lineStart = source.lastIndexOf("\n", statement.getStart() - 1) + 1;
    const indentation = source.slice(lineStart, statement.getStart());
    const declaration = `const ${variableName} = ${expression.getText()};\n${indentation}`;
    return {
      source: applyTextEdits(source, [
        {
          start: expression.getStart(),
          end: expression.getEnd(),
          text: variableName,
        },
        {
          start: statement.getStart(),
          end: statement.getStart(),
          text: declaration,
        },
      ]),
      changes: 1,
    };
  },

  replace_call(params) {
    const source = requireString(params, "source");
    const oldCall = requireString(params, "oldCall");
    const newCall = requireString(params, "newCall");
    const diagnostics = syntaxDiagnostics(
      `const __moray_probe = (${newCall});\n`,
      "probe.ts",
    );
    if (diagnostics.length > 0) {
      fail(
        INVALID_PARAMS,
        `newCall is not a valid expression: ${diagnostics[0].message}`,
      );
    }
    const project = new Project({ useInMemoryFileSystem: true });
    const sourceFile = project.createSourceFile("input.ts", source);
    const calls = sourceFile
      .getDescendantsOfKind(SyntaxKind.CallExpression)
      .filter((call) => call.getExpression().getText() === oldCall);
    if (calls.length === 0) {
      return { source, changes: 0 };
    }
    return {
      source: applyTextEdits(
        source,
        calls.map((call) => ({
          start: call.getExpression().getStart(),
          end: call.getExpression().getEnd(),
          text: newCall,
        })),
      ),
      changes: calls.length,
    };
  },

  add_import(params) {
    const source = requireString(params, "source");
    const moduleSpecifier = requireString(params, "module");
    const name = optionalString(params, "name");
    const alias = optionalString(params, "alias");
    if (name === null && alias !== null) {
      fail(INVALID_PARAMS, "alias requires name");
    }
    const project = new Project({ useInMemoryFileSystem: true });
    const sourceFile = project.createSourceFile("input.ts", source);
    const declarations = sourceFile
      .getImportDeclarations()
      .filter(
        (declaration) =>
          declaration.getModuleSpecifierValue() === moduleSpecifier,
      );

    if (name === null) {
      if (declarations.some((declaration) => !declaration.getImportClause())) {
        return { source, changes: 0 };
      }
      sourceFile.addImportDeclaration({ moduleSpecifier });
      return { source: sourceFile.getFullText(), changes: 1 };
    }

    for (const declaration of declarations) {
      if (
        declaration
          .getNamedImports()
          .some((named) => named.getName() === name)
      ) {
        return { source, changes: 0 };
      }
    }
    const mergeable = declarations.find(
      (declaration) =>
        !declaration.getNamespaceImport() &&
        !declaration.getDefaultImport() &&
        !declaration.getImportClause()?.isTypeOnly(),
    );
    if (mergeable !== undefined) {
      mergeable.addNamedImport(alias === null ? name : { name, alias });
    } else {
      sourceFile.addImportDeclaration({
        moduleSpecifier,
        namedImports: [alias === null ? { name } : { name, alias }],
      });
    }
    return { source: sourceFile.getFullText(), changes: 1 };
  },

  remove_import(params) {
    const source = requireString(params, "source");
    const moduleSpecifier = requireString(params, "module");
    const name = optionalString(params, "name");
    const project = new Project({ useInMemoryFileSystem: true });
    const sourceFile = project.createSourceFile("input.ts", source);
    const declarations = sourceFile
      .getImportDeclarations()
      .filter(
        (declaration) =>
          declaration.getModuleSpecifierValue() === moduleSpecifier,
      );
    if (declarations.length === 0) {
      return { source, changes: 0 };
    }

    if (name === null) {
      for (const declaration of declarations) {
        declaration.remove();
      }
      return { source: sourceFile.getFullText(), changes: declarations.length };
    }

    let changes = 0;
    for (const declaration of declarations) {
      const matches = declaration
        .getNamedImports()
        .filter((named) => named.getName() === name);
      if (matches.length === 0) continue;
      for (const match of matches) {
        match.remove();
      }
      changes += 1;
      if (
        declaration.getNamedImports().length === 0 &&
        !declaration.getDefaultImport() &&
        !declaration.getNamespaceImport()
      ) {
        declaration.remove();
      }
    }
    return { source: sourceFile.getFullText(), changes };
  },
};

function respond(id, result) {
  process.stdout.write(
    `${JSON.stringify({ jsonrpc: "2.0", id, result })}\n`,
  );
}

function respondError(id, code, message) {
  process.stdout.write(
    `${JSON.stringify({ jsonrpc: "2.0", id, error: { code, message } })}\n`,
  );
}

function main() {
  let request;
  try {
    const raw = fs.readFileSync(0, "utf8");
    const line = raw
      .split(/\r?\n/)
      .map((part) => part.trim())
      .find((part) => part.length > 0);
    if (line === undefined) {
      respondError(null, INVALID_PARAMS, "empty request");
      return;
    }
    request = JSON.parse(line);
  } catch {
    respondError(null, INVALID_PARAMS, "request is not valid JSON");
    return;
  }

  const id = request?.id ?? null;
  if (request?.jsonrpc !== "2.0") {
    respondError(id, INVALID_PARAMS, "jsonrpc must be '2.0'");
    return;
  }
  const handler = handlers[request.method];
  if (typeof handler !== "function") {
    respondError(id, -32601, `unknown method '${request.method}'`);
    return;
  }
  try {
    const result = handler(requireObject(request.params ?? {}));
    respond(id, result);
  } catch (error) {
    if (error instanceof BridgeError) {
      respondError(id, error.code, error.message);
      return;
    }
    const stack =
      typeof error?.stack === "string"
        ? error.stack.slice(0, MAX_STACK_CHARS)
        : String(error);
    respondError(id, INTERNAL_ERROR, stack);
  }
}

main();
