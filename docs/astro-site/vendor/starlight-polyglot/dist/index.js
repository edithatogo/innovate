// ../../node_modules/.pnpm/tsup@8.5.1_jiti@2.7.0_postcss@8.5.14_typescript@5.9.3_yaml@2.9.0/node_modules/tsup/assets/esm_shims.js
import path from "path";
import { fileURLToPath } from "url";
var getFilename = () => fileURLToPath(import.meta.url);
var getDirname = () => path.dirname(getFilename());
var __dirname = /* @__PURE__ */ getDirname();

// index.ts
import { randomBytes } from "crypto";

// core/plugin.ts
function getSidebarGroupPlaceholder(key) {
  return {
    _key: key ?? /* @__PURE__ */ Symbol("starlight-polyglot"),
    label: "API",
    items: []
  };
}

// handlers/cpp.ts
import { execSync } from "child_process";
import { existsSync, readFileSync, readdirSync, writeFileSync } from "fs";
import path3 from "path";

// core/mdx-generator.ts
import fs from "fs/promises";
import path2 from "path";
import { slug } from "github-slugger";
function transformToMDX(modules, options) {
  const pages = [];
  const sidebarItems = [];
  const { outputDir, language } = options;
  for (const mod of modules) {
    const modSlug = slug(mod.name);
    const modLink = `${outputDir}/${modSlug}/`;
    pages.push({
      path: `${outputDir}/${modSlug}.mdx`,
      frontmatter: {
        title: mod.name,
        description: mod.docstring?.split("\n")[0] ?? `${language ?? ""} module: ${mod.name}`,
        sidebar: { label: mod.name },
        pagefind: true,
        ...language ? { language } : {},
        source: mod.name
      },
      body: generateModuleBody(mod)
    });
    sidebarItems.push({ label: mod.name, link: modLink });
    for (const cls of mod.classes ?? []) {
      const clsSlug = `${modSlug}.${slug(cls.name)}`;
      pages.push({
        path: `${outputDir}/${clsSlug}.mdx`,
        frontmatter: {
          title: `${mod.name}.${cls.name}`,
          description: cls.docstring?.split("\n")[0] ?? `Class ${cls.name}`,
          sidebar: { label: cls.name },
          pagefind: true,
          ...language ? { language } : {},
          source: `${mod.name}.${cls.name}`
        },
        body: generateClassBody(cls)
      });
    }
    for (const fn of mod.functions ?? []) {
      const fnSlug = `${modSlug}.${slug(fn.name)}`;
      pages.push({
        path: `${outputDir}/${fnSlug}.mdx`,
        frontmatter: {
          title: `${mod.name}.${fn.name}`,
          description: fn.docstring?.split("\n")[0] ?? `Function ${fn.name}`,
          sidebar: { label: fn.name },
          pagefind: true,
          ...language ? { language } : {},
          source: `${mod.name}.${fn.name}`
        },
        body: generateFunctionBody(fn)
      });
    }
  }
  return {
    pages,
    sidebar: {
      label: (language ?? "API").toUpperCase(),
      items: sidebarItems
    }
  };
}
function generateModuleBody(mod) {
  const parts = [];
  if (mod.docstring) {
    parts.push(mod.docstring, "");
  }
  if (mod.classes && mod.classes.length > 0) {
    parts.push("## Classes", "");
    for (const cls of mod.classes) {
      parts.push(`- [${cls.name}](${slug(mod.name)}.${slug(cls.name)}/) ${cls.docstring?.split("\n")[0] ?? ""}`);
    }
    parts.push("");
  }
  if (mod.functions && mod.functions.length > 0) {
    parts.push("## Functions", "");
    for (const fn of mod.functions) {
      parts.push(`- [${fn.name}](${slug(mod.name)}.${slug(fn.name)}/)
  ${fn.docstring?.split("\n")[0] ?? ""}`);
    }
    parts.push("");
  }
  return parts.join("\n");
}
function generateClassBody(cls) {
  const parts = [];
  if (cls.docstring) {
    parts.push(cls.docstring, "");
  }
  if (cls.methods && cls.methods.length > 0) {
    parts.push("## Methods", "");
    for (const method of cls.methods) {
      parts.push(generateFunctionBody(method));
    }
  }
  if (cls.properties && cls.properties.length > 0) {
    parts.push("## Properties", "");
    for (const prop of cls.properties) {
      parts.push(`### ${prop.name}`);
      if (prop.type) parts.push(`- **Type**: \`${prop.type}\``);
      if (prop.docstring) parts.push(`- ${prop.docstring}`);
      parts.push("");
    }
  }
  return parts.join("\n");
}
function generateFunctionBody(fn) {
  const parts = [];
  parts.push(`### ${fn.name}`);
  if (fn.signature) {
    parts.push("", "```", fn.signature, "```", "");
  }
  if (fn.docstring) {
    parts.push(fn.docstring, "");
  }
  if (fn.parameters && fn.parameters.length > 0) {
    parts.push("**Parameters:**", "");
    for (const param of fn.parameters) {
      const defaultStr = param.default ? ` (default: \`${param.default}\`)` : "";
      const typeStr = param.type ? `\`${param.type}\`` : "";
      parts.push(`- \`${param.name}\`${typeStr ? ` ${typeStr}` : ""}${defaultStr}`);
      if (param.description) parts.push(`  - ${param.description}`);
    }
    parts.push("");
  }
  if (fn.return_type) {
    parts.push(`**Returns:** \`${fn.return_type}\``, "");
  }
  return parts.join("\n");
}

// handlers/cpp.ts
var cppHandler = {
  name: "cpp",
  async generate(options) {
    const opts = options;
    const projectPath = opts.projectPath;
    const doxyfilePath = opts.doxyfilePath;
    if (!projectPath) {
      throw new Error("C++ handler requires a projectPath option");
    }
    if (!existsSync(projectPath)) {
      throw new Error(`Project path does not exist: ${projectPath}`);
    }
    const modules = extractWithDoxygen(projectPath, doxyfilePath, opts.inputDirs);
    if (modules.length === 0) {
      throw new Error("Doxygen extraction produced no modules");
    }
    const output = transformToMDX(modules, {
      outputDir: opts.output,
      language: "cpp",
      ...opts.pagination !== void 0 ? { pagination: opts.pagination } : {}
    });
    return output;
  },
  async validate(_sourcePath) {
    try {
      execSync("doxygen --version", { encoding: "utf-8", stdio: "pipe" });
      return { valid: true, errors: [] };
    } catch {
      return {
        valid: false,
        errors: [
          "doxygen not found. Install from https://www.doxygen.nl/download.html"
        ]
      };
    }
  }
};
function extractWithDoxygen(projectPath, doxyfilePath, inputDirs) {
  const resolvedPath = path3.resolve(projectPath);
  const outputDir = path3.resolve(resolvedPath, "doxygen_xml");
  const doxyfile = doxyfilePath ? path3.resolve(doxyfilePath) : path3.resolve(resolvedPath, "Doxyfile");
  if (!existsSync(doxyfile)) {
    generateMinimalDoxyfile(resolvedPath, outputDir, inputDirs);
  }
  const doxygenCmd = `doxygen "${doxyfile}" 2>&1`;
  execSync(doxygenCmd, {
    encoding: "utf-8",
    cwd: resolvedPath,
    stdio: "pipe",
    timeout: 18e4
  });
  if (!existsSync(outputDir)) {
    throw new Error(
      `Doxygen XML output directory not found at ${outputDir}. Ensure Doxygen is configured with GENERATE_XML=YES.`
    );
  }
  return parseDoxygenXml(outputDir);
}
function generateMinimalDoxyfile(projectPath, xmlOutputDir, inputDirs) {
  const input = inputDirs && inputDirs.length > 0 ? inputDirs.map((d) => path3.resolve(projectPath, d)).join(" ") : projectPath;
  const doxyfileContent = [
    'PROJECT_NAME           = "starlight-polyglot-cpp"',
    "OUTPUT_DIRECTORY       = " + projectPath,
    "GENERATE_XML           = YES",
    "XML_OUTPUT             = doxygen_xml",
    "INPUT                  = " + input,
    "RECURSIVE              = YES",
    "EXTRACT_ALL            = YES",
    "EXTRACT_STATIC         = YES",
    "EXTRACT_PRIVATE        = YES",
    "EXTRACT_PACKAGE        = YES",
    "SOURCE_BROWSER         = NO",
    "HTML_OUTPUT            = /dev/null",
    "QUIET                  = YES",
    "WARNINGS               = NO"
  ].join("\n");
  writeFileSync(path3.join(projectPath, "Doxyfile"), doxyfileContent, "utf-8");
}
function parseDoxygenXml(xmlDir) {
  const modules = [];
  const files = readdirSync(xmlDir).filter(
    (f) => f.endsWith(".xml") && f !== "index.xml" && f !== "DoxygenLayout.xml"
  );
  for (const file of files) {
    const filePath = path3.join(xmlDir, file);
    const raw = readFileSync(filePath, "utf-8");
    try {
      const data = parseDoxygenXmlRaw(raw);
      if (data) {
        const mod = convertDoxygenCompound(data);
        if (mod) {
          modules.push(mod);
        }
      }
    } catch {
      continue;
    }
  }
  return modules;
}
function parseDoxygenXmlRaw(xmlContent) {
  const compoundDefMatch = xmlContent.match(
    /<compounddef\s+kind="([^"]*)"[^>]*>([\s\S]*?)<\/compounddef>/
  );
  if (!compoundDefMatch) return null;
  const kind = compoundDefMatch[1];
  const body = compoundDefMatch[2];
  const nameMatch = body.match(/<compoundname>([^<]*)<\/compoundname>/);
  const name = nameMatch ? nameMatch[1].trim() : void 0;
  const briefMatch = body.match(/<briefdescription>([\s\S]*?)<\/briefdescription>/);
  const briefdescription = extractParaText(briefMatch ? briefMatch[1] : "");
  const detailedMatch = body.match(/<detaileddescription>([\s\S]*?)<\/detaileddescription>/);
  const detaileddescription = extractParaText(detailedMatch ? detailedMatch[1] : "");
  const compound = {
    kind,
    name,
    briefdescription,
    detaileddescription,
    sectiondef: []
  };
  const sectionRegex = /<sectiondef\s+kind="([^"]*)">([\s\S]*?)<\/sectiondef>/g;
  let sectionMatch;
  while ((sectionMatch = sectionRegex.exec(body)) !== null) {
    const sectionKind = sectionMatch[1];
    const sectionBody = sectionMatch[2];
    const members = parseMemberDefs(sectionBody);
    compound.sectiondef?.push({
      kind: sectionKind,
      memberdef: members.length > 0 ? members : void 0
    });
  }
  return compound;
}
function extractParaText(paraContent) {
  const cleaned = paraContent.replace(/<[^>]*>/g, "").replace(/\s+/g, " ").trim();
  return cleaned || void 0;
}
function parseMemberDefs(sectionBody) {
  const members = [];
  const memberRegex = /<memberdef\s+kind="([^"]*)"[^>]*>([\s\S]*?)<\/memberdef>/g;
  function convertDoxygenCompound2(compound) {
    if (!compound.name) return null;
    const description = compound.detaileddescription ?? compound.briefdescription;
    const mod = {
      name: compound.name,
      docstring: description,
      classes: [],
      functions: [],
      variables: []
    };
    for (const section of compound.sectiondef ?? []) {
      for (const member of section.memberdef ?? []) {
        if (!member.name) continue;
        if (member.kind === "function" || member.kind === "signal" || member.kind === "slot") {
          mod.functions?.push({
            name: member.name,
            signature: `${member.definition ?? ""} ${member.argsstring ?? ""}`.trim() || void 0,
            docstring: member.detaileddescription ?? member.briefdescription ?? void 0,
            parameters: member.param?.map((p) => ({
              name: p.name ?? "param",
              type: p.type ?? void 0,
              description: p.briefdescription ?? void 0,
              default: p.defval ?? void 0
            })),
            return_type: member.type ?? void 0
          });
        } else if (member.kind === "variable" || member.kind === "enumvalue") {
          mod.variables?.push({
            name: member.name,
            type: member.type ?? void 0,
            docstring: member.detaileddescription ?? member.briefdescription ?? void 0
          });
        }
      }
    }
    if (compound.kind === "class" || compound.kind === "struct" || compound.kind === "union" || compound.kind === "interface") {
      const cls = {
        name: compound.name,
        docstring: description,
        methods: [],
        properties: []
      };
      for (const section of compound.sectiondef ?? []) {
        for (const member of section.memberdef ?? []) {
          if (!member.name) continue;
          if (member.kind === "function" || member.kind === "signal" || member.kind === "slot") {
            cls.methods?.push({
              name: member.name,
              signature: `${member.definition ?? ""} ${member.argsstring ?? ""}`.trim() || void 0,
              docstring: member.detaileddescription ?? member.briefdescription ?? void 0,
              parameters: member.param?.map((p) => ({
                name: p.name ?? "param",
                type: p.type ?? void 0,
                description: p.briefdescription ?? void 0,
                default: p.defval ?? void 0
              })),
              return_type: member.type ?? void 0
            });
          } else if (member.kind === "variable" || member.kind === "enumvalue") {
            cls.properties?.push({
              name: member.name,
              type: member.type ?? void 0,
              docstring: member.detaileddescription ?? member.briefdescription ?? void 0
            });
          }
        }
      }
      mod.functions = [];
      mod.classes?.push(cls);
    }
    return mod;
  }
  let match;
  while ((match = memberRegex.exec(sectionBody)) !== null) {
    const kind = match[1];
    const body = match[2];
    const nameMatch = body.match(/<name>([^<]*)<\/name>/);
    const name = nameMatch ? nameMatch[1].trim() : void 0;
    const defMatch = body.match(/<definition>([^<]*)<\/definition>/);
    const definition = defMatch ? defMatch[1].trim() : void 0;
    const argsMatch = body.match(/<argsstring>([^<]*)<\/argsstring>/);
    const argsstring = argsMatch ? argsMatch[1].trim() : void 0;
    const typeMatch = body.match(/<type>([\s\S]*?)<\/type>/);
    const type = typeMatch ? typeMatch[1].replace(/<[^>]*>/g, "").trim() : void 0;
    const briefMatch = body.match(/<briefdescription>([\s\S]*?)<\/briefdescription>/);
    const briefdescription = extractParaText(briefMatch ? briefMatch[1] : "");
    const detailedMatch = body.match(/<detaileddescription>([\s\S]*?)<\/detaileddescription>/);
    const detaileddescription = extractParaText(detailedMatch ? detailedMatch[1] : "");
    const initMatch = body.match(/<initializer>([^<]*)<\/initializer>/);
    const initializer = initMatch ? initMatch[1].trim() : void 0;
    const params = [];
    const paramRegex = /<param>([\s\S]*?)<\/param>/g;
    let paramMatch;
    while ((paramMatch = paramRegex.exec(body)) !== null) {
      const pBody = paramMatch[1];
      const pName = pBody.match(/<declname>([^<]*)<\/declname>/)?.[1]?.trim();
      const pType = pBody.match(/<type>([\s\S]*?)<\/type>/)?.[1]?.replace(/<[^>]*>/g, "").trim();
      const pDefval = pBody.match(/<defval>([^<]*)<\/defval>/)?.[1]?.trim();
      const pBrief = extractParaText(
        pBody.match(/<briefdescription>([\s\S]*?)<\/briefdescription>/)?.[1] ?? ""
      );
      params.push({ name: pName, type: pType, defval: pDefval, briefdescription: pBrief });
    }
    members.push({
      kind,
      name,
      definition,
      argsstring,
      briefdescription,
      detaileddescription,
      type,
      initializer,
      param: params.length > 0 ? params : void 0
    });
  }
  return members;
}

// handlers/csharp.ts
import { execSync as execSync2 } from "child_process";
import { existsSync as existsSync2, readFileSync as readFileSync2, readdirSync as readdirSync2 } from "fs";
import path4 from "path";
var csharpHandler = {
  name: "csharp",
  async generate(options) {
    const opts = options;
    const projectPath = opts.projectPath;
    if (!projectPath) {
      throw new Error("C# handler requires a projectPath option");
    }
    if (!existsSync2(projectPath)) {
      throw new Error(`Project path does not exist: ${projectPath}`);
    }
    const modules = extractWithDotNet(projectPath, opts.configuration, opts.xmlDocPath);
    if (modules.length === 0) {
      throw new Error("C# XML doc extraction produced no modules");
    }
    const output = transformToMDX(modules, {
      outputDir: opts.output,
      language: "csharp",
      ...opts.pagination !== void 0 ? { pagination: opts.pagination } : {}
    });
    return output;
  },
  async validate(_sourcePath) {
    try {
      execSync2("dotnet --version", { encoding: "utf-8", stdio: "pipe" });
      return { valid: true, errors: [] };
    } catch {
      return { valid: false, errors: ["dotnet SDK not found. Install from https://dotnet.microsoft.com/download"] };
    }
  }
};
function extractWithDotNet(projectPath, configuration = "Release", explicitDocPath) {
  const resolvedPath = path4.resolve(projectPath);
  const buildCmd = `dotnet build "${resolvedPath}" --configuration ${configuration} /p:GenerateDocumentationFile=true 2>&1`;
  execSync2(buildCmd, {
    encoding: "utf-8",
    stdio: "pipe",
    timeout: 18e4
  });
  let xmlDocPath = explicitDocPath;
  if (!xmlDocPath) {
    const projectName = path4.basename(resolvedPath, path4.extname(resolvedPath));
    const possiblePaths = [
      path4.resolve(resolvedPath, "bin", configuration, "net9.0", `${projectName}.xml`),
      path4.resolve(resolvedPath, "bin", configuration, "net8.0", `${projectName}.xml`),
      path4.resolve(resolvedPath, "bin", configuration, "net7.0", `${projectName}.xml`),
      path4.resolve(resolvedPath, "bin", configuration, "net6.0", `${projectName}.xml`),
      path4.resolve(resolvedPath, "bin", configuration, "netstandard2.0", `${projectName}.xml`)
    ];
    const binDir = path4.resolve(resolvedPath, "bin", configuration);
    if (existsSync2(binDir)) {
      try {
        const frameworks = readdirSync2(binDir, { withFileTypes: true });
        for (const fw of frameworks) {
          if (fw.isDirectory()) {
            possiblePaths.push(path4.resolve(binDir, fw.name, `${projectName}.xml`));
          }
        }
      } catch {
      }
    }
    for (const p of possiblePaths) {
      if (existsSync2(p)) {
        xmlDocPath = p;
        break;
      }
    }
  }
  if (!xmlDocPath || !existsSync2(xmlDocPath)) {
    throw new Error(
      "Could not find XML documentation file. Ensure GenerateDocumentationFile is enabled in the .csproj."
    );
  }
  return parseXmlDocFile(xmlDocPath);
}
function parseXmlDocFile(xmlPath) {
  const xmlContent = readFileSync2(xmlPath, "utf-8");
  const modules = [];
  const memberRegex = /<member\s+name="([^"]+)"\s*>([\s\S]*?)<\/member>/g;
  const summaryRegex = /<summary>([\s\S]*?)<\/summary>/;
  const paramRegex = /<param\s+name="([^"]+)"\s*>([\s\S]*?)<\/param>/g;
  const returnsRegex = /<returns>([\s\S]*?)<\/returns>/;
  const members = [];
  let match;
  while ((match = memberRegex.exec(xmlContent)) !== null) {
    const memberName = match[1];
    const memberBody = match[2];
    const summary = summaryRegex.exec(memberBody)?.[1]?.trim();
    const returns = returnsRegex.exec(memberBody)?.[1]?.trim();
    const params = [];
    let paramMatch;
    const localParamRegex = new RegExp(paramRegex.source, "g");
    while ((paramMatch = localParamRegex.exec(memberBody)) !== null) {
      params.push({ name: paramMatch[1], text: paramMatch[2].trim() });
    }
    members.push({
      name: memberName,
      summary: summary?.trim() || void 0,
      params: params.length > 0 ? params : void 0,
      returns: returns?.trim() || void 0
    });
  }
  const moduleMap = /* @__PURE__ */ new Map();
  const typeMap = /* @__PURE__ */ new Map();
  for (const member of members) {
    const parts = member.name.split(":");
    if (parts.length < 2) continue;
    const prefix = parts[0];
    const fullName = parts.slice(1).join(":");
    if (prefix === "T") {
      const lastDot = fullName.lastIndexOf(".");
      const namespace = lastDot >= 0 ? fullName.substring(0, lastDot) : "";
      const typeName = lastDot >= 0 ? fullName.substring(lastDot + 1) : fullName;
      if (!moduleMap.has(namespace)) {
        moduleMap.set(namespace, {
          name: namespace || "Global",
          docstring: void 0,
          classes: [],
          functions: [],
          variables: []
        });
      }
      typeMap.set(fullName, {
        name: typeName,
        docstring: member.summary,
        methods: [],
        properties: []
      });
    } else if (prefix === "M") {
      const parenIndex = fullName.indexOf("(");
      const methodPath = parenIndex >= 0 ? fullName.substring(0, parenIndex) : fullName;
      const lastDot = methodPath.lastIndexOf(".");
      const parentType = lastDot >= 0 ? methodPath.substring(0, lastDot) : "";
      const methodName = lastDot >= 0 ? methodPath.substring(lastDot + 1) : methodPath;
      const typeEntry = typeMap.get(parentType);
      if (typeEntry) {
        typeEntry.methods.push({
          name: methodName,
          docstring: member.summary,
          parameters: member.params?.map((p) => ({
            name: p.name,
            type: void 0,
            description: p.text || void 0,
            default: void 0
          })),
          return_type: member.returns
        });
      }
    } else if (prefix === "P" || prefix === "F") {
      const lastDot = fullName.lastIndexOf(".");
      const parentType = lastDot >= 0 ? fullName.substring(0, lastDot) : "";
      const propName = lastDot >= 0 ? fullName.substring(lastDot + 1) : fullName;
      const typeEntry = typeMap.get(parentType);
      if (typeEntry) {
        typeEntry.properties.push({
          name: propName,
          docstring: member.summary
        });
      }
    }
  }
  for (const [fullName, typeEntry] of typeMap) {
    const lastDot = fullName.lastIndexOf(".");
    const namespace = lastDot >= 0 ? fullName.substring(0, lastDot) : "";
    const mod = moduleMap.get(namespace);
    if (mod) {
      mod.classes?.push({
        name: typeEntry.name,
        docstring: typeEntry.docstring,
        methods: typeEntry.methods.length > 0 ? typeEntry.methods : void 0,
        properties: typeEntry.properties.length > 0 ? typeEntry.properties : void 0
      });
    }
  }
  return Array.from(moduleMap.values());
}

// handlers/dart.ts
import { execSync as execSync3 } from "child_process";
import { existsSync as existsSync3, readFileSync as readFileSync3 } from "fs";
import path5 from "path";
var dartHandler = {
  name: "dart",
  async generate(options) {
    const opts = options;
    const entryPoints = opts.entryPoints;
    if (!entryPoints || entryPoints.length === 0) {
      throw new Error("Dart handler requires at least one entryPoint");
    }
    const modules = extractWithDartdoc(entryPoints);
    if (modules.length === 0) {
      throw new Error("dartdoc extraction produced no modules");
    }
    const output = transformToMDX(modules, {
      outputDir: opts.output,
      language: "dart",
      ...opts.pagination !== void 0 ? { pagination: opts.pagination } : {}
    });
    return output;
  },
  async validate(_sourcePath) {
    try {
      execSync3("dart --version", { encoding: "utf-8", stdio: "pipe" });
      execSync3("dart doc --help", { encoding: "utf-8", stdio: "pipe" });
      return { valid: true, errors: [] };
    } catch {
      return {
        valid: false,
        errors: [
          "Dart SDK not found. Install from https://dart.dev/get-dart"
        ]
      };
    }
  }
};
function extractWithDartdoc(entryPoints) {
  const modules = [];
  for (const entry of entryPoints) {
    const resolvedEntry = path5.resolve(entry);
    if (!existsSync3(resolvedEntry)) {
      continue;
    }
    const jsonOutputPath = "/tmp/starlight-polyglot-dartdoc.json";
    const cmd = `dart doc --json-output "${jsonOutputPath}" --output /tmp/starlight-polyglot-dartdoc-html "${resolvedEntry}" 2>&1`;
    execSync3(cmd, {
      encoding: "utf-8",
      cwd: existsSync3(path5.join(resolvedEntry, "pubspec.yaml")) ? resolvedEntry : path5.dirname(resolvedEntry),
      maxBuffer: 10 * 1024 * 1024,
      timeout: 18e4,
      stdio: "pipe"
    });
    if (existsSync3(jsonOutputPath)) {
      const raw = readFileSync3(jsonOutputPath, "utf-8");
      try {
        const dartdocData = JSON.parse(raw);
        const converted = convertDartdocElements(dartdocData);
        modules.push(...converted);
      } catch {
        continue;
      }
    }
  }
  return modules;
  function convertDartdocElements(elements) {
    const modules2 = [];
    const libraryMap = /* @__PURE__ */ new Map();
    for (const el of elements) {
      const libName = el.enclosingElement?.qualifiedName ?? el.qualifiedName?.split(".").shift() ?? "Global";
      if (!libraryMap.has(libName)) {
        libraryMap.set(libName, []);
      }
      libraryMap.get(libName).push(el);
    }
    for (const [libName, libElements] of libraryMap) {
      const mod = {
        name: libName,
        docstring: void 0,
        classes: [],
        functions: [],
        variables: []
      };
      const classes = libElements.filter(
        (e) => (e.kind === "class" || e.kind === "mixin" || e.kind === "enum" || e.kind === "extension") && (!e.enclosingElement || !e.enclosingElement.name)
      );
      for (const cls of classes) {
        const clsDoc = extractDartdocDoc(cls);
        const clsResult = {
          name: cls.name ?? "Unknown",
          docstring: clsDoc,
          methods: [],
          properties: []
        };
        const members = libElements.filter(
          (e) => e.enclosingElement?.name === cls.name
        );
        for (const member of members) {
          const memberDoc = extractDartdocDoc(member);
          if (member.kind === "method" || member.kind === "constructor" || member.kind === "method_operator") {
            clsResult.methods.push({
              name: member.name ?? "unknown",
              signature: buildDartSignature(member),
              docstring: memberDoc,
              parameters: (member.parameters ?? []).map((p) => ({
                name: p.name,
                type: p.type ?? void 0,
                description: void 0,
                default: p.defaultValue ?? void 0
              })),
              return_type: member.returnType ?? void 0
            });
          } else if (member.kind === "property" || member.kind === "field" || member.kind === "constant") {
            clsResult.properties.push({
              name: member.name ?? "unknown",
              type: member.returnType ?? member.type ?? void 0,
              docstring: memberDoc
            });
          }
        }
        mod.classes?.push(clsResult);
        if (!mod.docstring && clsDoc) {
          mod.docstring = clsDoc;
        }
      }
      const functions = libElements.filter(
        (e) => (e.kind === "function" || e.kind === "top_level_function" || e.kind === "method") && (!e.enclosingElement || !e.enclosingElement.name)
      );
      for (const fn of functions) {
        mod.functions?.push({
          name: fn.name ?? "unknown",
          signature: buildDartSignature(fn),
          docstring: extractDartdocDoc(fn),
          parameters: (fn.parameters ?? []).map((p) => ({
            name: p.name,
            type: p.type ?? void 0,
            description: void 0,
            default: p.defaultValue ?? void 0
          })),
          return_type: fn.returnType ?? void 0
        });
      }
      const variables = libElements.filter(
        (e) => (e.kind === "constant" || e.kind === "top_level_variable" || e.kind === "variable") && (!e.enclosingElement || !e.enclosingElement.name)
      );
      for (const v of variables) {
        mod.variables?.push({
          name: v.name ?? "unknown",
          type: v.returnType ?? v.type ?? void 0,
          docstring: extractDartdocDoc(v)
        });
      }
      if (mod.classes && mod.classes.length > 0 || mod.functions && mod.functions.length > 0 || mod.variables && mod.variables.length > 0) {
        modules2.push(mod);
      }
    }
    return modules2;
  }
  function extractDartdocDoc(el) {
    return el.documentation?.trim() || el.description?.trim() || void 0;
  }
  function buildDartSignature(el) {
    const params = (el.parameters ?? []).map((p) => {
      const typeStr = p.type ? ` ${p.type}` : "";
      const optional = p.isOptional ? "?" : "";
      return `${p.name}${optional}:${typeStr}`;
    }).join(", ");
    const returnType = el.returnType ? ` \u2192 ${el.returnType}` : "";
    return `${el.name}(${params})${returnType}`;
  }
}

// handlers/elixir.ts
import { execSync as execSync4 } from "child_process";
import { existsSync as existsSync4, readFileSync as readFileSync4, readdirSync as readdirSync3 } from "fs";
import path6 from "path";
var elixirHandler = {
  name: "elixir",
  async generate(options) {
    const opts = options;
    const projectPath = opts.projectPath;
    if (!projectPath) {
      throw new Error("Elixir handler requires a projectPath option");
    }
    if (!existsSync4(projectPath)) {
      throw new Error(`Project path does not exist: ${projectPath}`);
    }
    const mixExsPath = path6.join(projectPath, "mix.exs");
    if (!existsSync4(mixExsPath)) {
      throw new Error(`No mix.exs found at ${projectPath}. Ensure the path is an Elixir Mix project root.`);
    }
    const modules = extractWithExDoc(projectPath);
    if (modules.length === 0) {
      throw new Error("ExDoc extraction produced no modules");
    }
    const output = transformToMDX(modules, {
      outputDir: opts.output,
      language: "elixir",
      ...opts.pagination !== void 0 ? { pagination: opts.pagination } : {}
    });
    return output;
  },
  async validate(_sourcePath) {
    try {
      execSync4("elixir --version", { encoding: "utf-8", stdio: "pipe" });
      execSync4("mix --version", { encoding: "utf-8", stdio: "pipe" });
      return { valid: true, errors: [] };
    } catch {
      return {
        valid: false,
        errors: [
          "Elixir not found. Install from https://elixir-lang.org/install.html"
        ]
      };
    }
  }
};
function extractWithExDoc(projectPath) {
  const resolvedPath = path6.resolve(projectPath);
  execSync4("mix deps.get", {
    encoding: "utf-8",
    cwd: resolvedPath,
    stdio: "pipe",
    timeout: 12e4
  });
  const cmd = "mix docs --formatter json 2>&1";
  execSync4(cmd, {
    encoding: "utf-8",
    cwd: resolvedPath,
    maxBuffer: 10 * 1024 * 1024,
    timeout: 18e4,
    stdio: "pipe"
  });
  const docDir = path6.join(resolvedPath, "doc");
  const jsonPaths = [
    path6.join(docDir, "ex_doc.json"),
    path6.join(docDir, "docs.json"),
    ...existsSync4(docDir) ? readdirSync3(docDir).filter((f) => f.endsWith(".json")).map((f) => path6.join(docDir, f)) : []
  ];
  for (const jsonPath of jsonPaths) {
    if (existsSync4(jsonPath)) {
      const raw = readFileSync4(jsonPath, "utf-8");
      try {
        const exDocData = JSON.parse(raw);
        return convertExDocModules(exDocData.modules ?? []);
      } catch {
        continue;
      }
    }
  }
  throw new Error(
    "ExDoc did not produce JSON output. Ensure ex_doc is configured in mix.exs and the project compiles."
  );
}
function convertExDocModules(exDocModules) {
  const modules = [];
  for (const exMod of exDocModules) {
    const mod = {
      name: exMod.module ?? exMod.id ?? "Unknown",
      docstring: exMod.moduledoc?.trim() || void 0,
      classes: [],
      functions: [],
      variables: []
    };
    for (const fn of exMod.functions ?? []) {
      const params = parseExDocSignature(fn.spec ?? fn.signature);
      mod.functions?.push({
        name: fn.name,
        signature: fn.spec ?? fn.signature ?? void 0,
        docstring: fn.doc?.trim() || void 0,
        parameters: params.length > 0 ? params : void 0,
        return_type: extractExDocReturnType(fn.spec ?? fn.signature)
      });
    }
    for (const cb of exMod.callbacks ?? []) {
      const params = parseExDocSignature(cb.spec ?? cb.signature);
      mod.functions?.push({
        name: cb.name,
        signature: cb.spec ?? cb.signature ?? `callback ${cb.name}`,
        docstring: cb.doc?.trim() || void 0,
        parameters: params.length > 0 ? params : void 0,
        return_type: extractExDocReturnType(cb.spec ?? cb.signature)
      });
    }
    for (const tp of exMod.types ?? []) {
      mod.variables?.push({
        name: tp.name,
        type: tp.spec ?? tp.type ?? "type",
        docstring: tp.doc?.trim() || void 0
      });
    }
    for (const task of exMod.tasks ?? []) {
      mod.functions?.push({
        name: task.name,
        signature: task.signature ?? task.spec ?? void 0,
        docstring: task.doc?.trim() || void 0,
        parameters: parseExDocSignature(task.signature),
        return_type: void 0
      });
    }
    modules.push(mod);
  }
  return modules;
}
function parseExDocSignature(sig) {
  if (!sig) return void 0;
  let depth = 0;
  let startIdx = -1;
  let endIdx = -1;
  for (let i = 0; i < sig.length; i++) {
    if (sig[i] === "(") {
      if (depth === 0) startIdx = i;
      depth++;
    } else if (sig[i] === ")") {
      depth--;
      if (depth === 0 && startIdx >= 0) {
        endIdx = i;
        break;
      }
    }
  }
  if (startIdx < 0 || endIdx < 0) return void 0;
  const paramsStr = sig.substring(startIdx + 1, endIdx);
  if (!paramsStr.trim()) return void 0;
  return paramsStr.split(",").map((p) => {
    const trimmed = p.trim();
    const parts = trimmed.split(/::|\\s+/);
    return {
      name: parts[0]?.trim() || trimmed,
      type: parts[1]?.trim() || void 0,
      description: void 0,
      default: void 0
    };
  });
}
function extractExDocReturnType(sig) {
  if (!sig) return void 0;
  const idx = sig.indexOf("::");
  if (idx < 0) return void 0;
  return sig.substring(idx + 2).trim() || void 0;
}

// handlers/go.ts
import { execSync as execSync5 } from "child_process";
import { existsSync as existsSync5 } from "fs";
import path7 from "path";
var goHandler = {
  name: "go",
  async generate(options) {
    const opts = options;
    const modulePath = opts.modulePath;
    if (!modulePath) {
      throw new Error("Go handler requires a modulePath option");
    }
    if (!existsSync5(modulePath)) {
      throw new Error(`Module path does not exist: ${modulePath}`);
    }
    const modules = extractWithGoMarkdoc(modulePath);
    if (modules.length === 0) {
      throw new Error("gomarkdoc extraction produced no modules");
    }
    const output = transformToMDX(modules, {
      outputDir: opts.output,
      language: "go",
      ...opts.pagination !== void 0 ? { pagination: opts.pagination } : {}
    });
    return output;
  },
  async validate(_sourcePath) {
    try {
      execSync5("gomarkdoc --version", { encoding: "utf-8", stdio: "pipe" });
      return { valid: true, errors: [] };
    } catch {
      return { valid: false, errors: ["gomarkdoc not found. Install: go install github.com/princjef/gomarkdoc/cmd/gomarkdoc@latest"] };
    }
  }
};
function extractWithGoMarkdoc(modulePath) {
  const resolvedPath = path7.resolve(modulePath);
  const cmd = `gomarkdoc --output json ./...`;
  const result = execSync5(cmd, {
    encoding: "utf-8",
    cwd: resolvedPath,
    maxBuffer: 10 * 1024 * 1024,
    // 10MB
    timeout: 12e4
  });
  const parsed = JSON.parse(result);
  const modules = [];
  for (const [pkgPath, pkgDoc] of Object.entries(parsed)) {
    const mod = {
      name: pkgDoc.name ?? pkgPath.split("/").pop() ?? pkgPath,
      docstring: pkgDoc.description || void 0,
      classes: [],
      functions: [],
      variables: []
    };
    for (const fn of pkgDoc.functions ?? []) {
      mod.functions?.push({
        name: fn.name,
        signature: fn.signature ?? void 0,
        docstring: fn.description || void 0,
        parameters: fn.params?.map((p) => ({
          name: p.name,
          type: p.type,
          description: void 0,
          default: void 0
        })),
        return_type: fn.returns?.map((r) => r.type).join(", ") || void 0
      });
    }
    for (const typ of pkgDoc.types ?? []) {
      mod.classes?.push({
        name: typ.name,
        docstring: typ.description || void 0,
        methods: typ.methods?.map((m) => ({
          name: m.name,
          signature: m.signature ?? void 0,
          docstring: m.description || void 0,
          parameters: m.params?.map((p) => ({
            name: p.name,
            type: p.type,
            description: void 0,
            default: void 0
          })),
          return_type: m.returns?.map((r) => r.type).join(", ") || void 0
        })),
        properties: typ.fields?.map((f) => ({
          name: f.name,
          type: f.type,
          docstring: f.description || void 0
        }))
      });
    }
    modules.push(mod);
  }
  return modules;
}

// handlers/java.ts
import { execSync as execSync6 } from "child_process";
import { existsSync as existsSync6, mkdtempSync, readFileSync as readFileSync5, readdirSync as readdirSync4 } from "fs";
import path8 from "path";
import { tmpdir } from "os";
var javaHandler = {
  name: "java",
  async generate(options) {
    const opts = options;
    const entryPoints = opts.entryPoints;
    const classpath = opts.classpath;
    if (!entryPoints || entryPoints.length === 0) {
      throw new Error("Java handler requires at least one entryPoint (source directory or package)");
    }
    const modules = extractWithJavadoc(entryPoints, classpath);
    if (modules.length === 0) {
      throw new Error("javadoc extraction produced no modules");
    }
    const output = transformToMDX(modules, {
      outputDir: opts.output,
      language: "java",
      ...opts.pagination !== void 0 ? { pagination: opts.pagination } : {}
    });
    return output;
  },
  async validate(_sourcePath) {
    try {
      execSync6("javadoc --version", { encoding: "utf-8", stdio: "pipe" });
      return { valid: true, errors: [] };
    } catch {
      return {
        valid: false,
        errors: [
          "javadoc not found. Ensure a JDK is installed and javadoc is on your PATH."
        ]
      };
    }
  }
};
function extractWithJavadoc(entryPoints, classpath) {
  const tmpDir = mkdtempSync(path8.join(tmpdir(), "javadoc-json-"));
  try {
    const cmdParts = ["javadoc", "-Xdoclint:none", "-json", "-d", tmpDir];
    if (classpath) {
      cmdParts.push("-classpath", classpath);
    }
    for (const entry of entryPoints) {
      const resolved = path8.resolve(entry);
      if (existsSync6(resolved)) {
        cmdParts.push(resolved);
      } else {
        cmdParts.push(entry);
      }
    }
    const cmd = cmdParts.join(" ");
    execSync6(cmd, {
      encoding: "utf-8",
      stdio: "pipe",
      maxBuffer: 10 * 1024 * 1024,
      timeout: 12e4
    });
    const files = readdirSync4(tmpDir);
    const jsonFiles = files.filter((f) => f.endsWith(".json"));
    if (jsonFiles.length === 0) {
      throw new Error(
        "javadoc did not produce any JSON output. Check that your JDK supports the -json flag."
      );
    }
    const modules = [];
    for (const jsonFile of jsonFiles) {
      const jsonPath = path8.join(tmpDir, jsonFile);
      const raw = readFileSync5(jsonPath, "utf-8");
      const parsed = JSON.parse(raw);
      for (const pkg of parsed.packages ?? []) {
        const mod = convertJavadocPackage(pkg);
        if (mod) {
          modules.push(mod);
        }
      }
      for (const cls of parsed.classes ?? []) {
        const mod = convertJavadocClassToModule(cls);
        if (mod) {
          modules.push(mod);
        }
      }
    }
    return modules;
  } finally {
    try {
      execSync6(`rm -rf "${tmpDir}"`, { stdio: "pipe" });
    } catch {
    }
  }
}
function convertJavadocPackage(pkg) {
  if (!pkg.name && !pkg.qualifiedName) return null;
  const mod = {
    name: pkg.qualifiedName ?? pkg.name ?? "unknown",
    docstring: pkg.comment?.trim() || void 0,
    classes: [],
    functions: [],
    variables: []
  };
  for (const member of pkg.members ?? []) {
    if (member.name || member.qualifiedName) {
      const cls = convertJavadocClass(member);
      if (cls) {
        mod.classes?.push(cls);
      }
    }
  }
  return mod;
}
function convertJavadocClassToModule(cls) {
  if (!cls.name && !cls.qualifiedName) return null;
  const innerClass = convertJavadocClass(cls);
  if (!innerClass) return null;
  return {
    name: cls.qualifiedName ?? cls.name ?? "unknown",
    docstring: cls.comment?.trim() || void 0,
    classes: [innerClass],
    functions: [],
    variables: []
  };
}
function convertJavadocClass(element) {
  const name = element.name ?? element.qualifiedName;
  if (!name) return null;
  const cls = {
    name,
    docstring: element.comment?.trim() || void 0,
    methods: [],
    properties: []
  };
  for (const member of element.members ?? []) {
    if (!member.name) continue;
    const isMethod = member.signature !== void 0 || member.params !== void 0 || member.return !== void 0;
    if (isMethod) {
      cls.methods.push({
        name: member.name,
        signature: buildJavaSignature(member),
        docstring: member.comment?.trim() || void 0,
        parameters: member.params?.map((p) => ({
          name: p.name,
          type: p.type ?? void 0,
          description: p.comment?.trim() || void 0,
          default: void 0
        })),
        return_type: member.return?.type ?? void 0
      });
    } else {
      cls.properties.push({
        name: member.name,
        type: member.return?.type ?? void 0,
        docstring: member.comment?.trim() || void 0
      });
    }
  }
  return cls;
}
function buildJavaSignature(element) {
  const params = (element.params ?? []).map((p) => `${p.type ?? "Object"} ${p.name}`).join(", ");
  const returnType = element.return?.type ?? "void";
  const modifiers = (element.modifiers ?? []).filter(
    (m) => m !== "abstract" && m !== "default"
  );
  const prefix = modifiers.length > 0 ? `${modifiers.join(" ")} ` : "";
  return `${prefix}${returnType} ${element.name}(${params})`;
}

// handlers/julia.ts
import { execSync as execSync7 } from "child_process";
import { existsSync as existsSync7 } from "fs";
import path9 from "path";
var juliaHandler = {
  name: "julia",
  async generate(options) {
    const opts = options;
    const entryPoints = opts.entryPoints;
    if (!entryPoints || entryPoints.length === 0) {
      throw new Error("Julia handler requires at least one entryPoint");
    }
    const ast = await extractWithJulia(entryPoints);
    const modules = ast.modules ?? [];
    if (modules.length === 0 && ast.errors) {
      throw new Error(
        `Julia extraction failed: ${ast.errors.map((e) => e.error).join(", ")}`
      );
    }
    const output = transformToMDX(modules, {
      outputDir: opts.output,
      language: "julia",
      ...opts.pagination !== void 0 ? { pagination: opts.pagination } : {}
    });
    return output;
  },
  async validate(_sourcePath) {
    try {
      execSync7("julia --version", { encoding: "utf-8", stdio: "pipe" });
      return { valid: true, errors: [] };
    } catch {
      return { valid: false, errors: ["julia not found. Install Julia from https://julialang.org/"] };
    }
  }
};
function extractWithJulia(entryPoints) {
  const scriptPath = path9.resolve(__dirname, "..", "scripts", "julia_extract.jl");
  if (!existsSync7(scriptPath)) {
    throw new Error(`Julia extraction script not found at ${scriptPath}`);
  }
  const args = ["julia", scriptPath, ...entryPoints];
  const result = execSync7(args.join(" "), {
    encoding: "utf-8",
    maxBuffer: 10 * 1024 * 1024,
    // 10MB
    timeout: 12e4
  });
  return JSON.parse(result);
}

// handlers/kotlin.ts
import { execSync as execSync8 } from "child_process";
import { existsSync as existsSync8, readFileSync as readFileSync6, readdirSync as readdirSync5 } from "fs";
import path10 from "path";
var kotlinHandler = {
  name: "kotlin",
  async generate(options) {
    const opts = options;
    const projectPath = opts.projectPath;
    if (!projectPath) {
      throw new Error("Kotlin handler requires a projectPath option");
    }
    if (!existsSync8(projectPath)) {
      throw new Error(`Project path does not exist: ${projectPath}`);
    }
    const modules = extractWithDokka(projectPath, opts.outputFormat);
    if (modules.length === 0) {
      throw new Error("Dokka extraction produced no modules");
    }
    const output = transformToMDX(modules, {
      outputDir: opts.output,
      language: "kotlin",
      ...opts.pagination !== void 0 ? { pagination: opts.pagination } : {}
    });
    return output;
  },
  async validate(_sourcePath) {
    try {
      execSync8("dokka --version", { encoding: "utf-8", stdio: "pipe" });
      return { valid: true, errors: [] };
    } catch {
      try {
        execSync8("gradle --version", { encoding: "utf-8", stdio: "pipe" });
        return { valid: true, errors: [] };
      } catch {
        return {
          valid: false,
          errors: [
            "Neither Dokka CLI nor Gradle found. Install Dokka from https://github.com/Kotlin/dokka or use Gradle."
          ]
        };
      }
    }
  }
};
function extractWithDokka(projectPath, _outputFormat) {
  const resolvedPath = path10.resolve(projectPath);
  const outputDir = path10.resolve(resolvedPath, "build", "dokka", "json");
  const gradleBuildFiles = ["build.gradle.kts", "build.gradle", "pom.xml"];
  const hasGradle = gradleBuildFiles.some(
    (f) => existsSync8(path10.resolve(resolvedPath, f))
  );
  if (hasGradle) {
    runGradleDokka(resolvedPath);
  } else {
    runDokkaCli(resolvedPath, outputDir);
  }
  if (!existsSync8(outputDir)) {
    throw new Error(
      `Dokka output directory not found at ${outputDir}. Ensure Dokka is configured to produce JSON output.`
    );
  }
  return parseDokkaOutput(outputDir);
}
function runGradleDokka(projectPath) {
  const gradleCmd = existsSync8(path10.join(projectPath, "gradlew")) ? "./gradlew" : "gradle";
  const cmd = `${gradleCmd} dokkaJson --no-daemon 2>&1`;
  execSync8(cmd, {
    encoding: "utf-8",
    cwd: projectPath,
    stdio: "pipe",
    timeout: 3e5
  });
}
function runDokkaCli(projectPath, outputDir) {
  const sourceDirs = findKotlinSourceDirs(projectPath);
  if (sourceDirs.length === 0) {
    throw new Error(
      `No Kotlin source directories found under ${projectPath}. Ensure your project has src/main/kotlin or similar.`
    );
  }
  const sourceArgs = sourceDirs.map((d) => `-src "${d}"`).join(" ");
  const cmd = `dokka -outputDir "${outputDir}" -format json ${sourceArgs} 2>&1`;
  execSync8(cmd, {
    encoding: "utf-8",
    cwd: projectPath,
    stdio: "pipe",
    timeout: 18e4
  });
}
function findKotlinSourceDirs(projectPath) {
  const candidates = [
    "src/main/kotlin",
    "src/commonMain/kotlin",
    "src/jvmMain/kotlin",
    "src/jsMain/kotlin",
    "src"
  ];
  const dirs = [];
  for (const candidate of candidates) {
    const fullPath = path10.resolve(projectPath, candidate);
    if (existsSync8(fullPath)) {
      dirs.push(fullPath);
    }
  }
  return dirs;
}
function parseDokkaOutput(outputDir) {
  const modules = [];
  const files = readdirSync5(outputDir).filter((f) => f.endsWith(".json"));
  for (const file of files) {
    const filePath = path10.join(outputDir, file);
    const raw = readFileSync6(filePath, "utf-8");
    const data = JSON.parse(raw);
    const moduleName = data.module ?? path10.basename(file, ".json");
    const mod = {
      name: moduleName,
      docstring: void 0,
      classes: [],
      functions: [],
      variables: []
    };
    for (const node of data.documentation ?? []) {
      convertDokkaNode(node, mod);
    }
    modules.push(mod);
  }
  return modules;
}
function convertDokkaNode(node, mod) {
  if (!node.name) return;
  const kind = node.kind ?? "";
  if (kind === "class" || kind === "interface" || kind === "object" || kind === "enum") {
    const cls = {
      name: node.name,
      docstring: node.description?.trim() || void 0,
      methods: [],
      properties: []
    };
    for (const child of node.children ?? []) {
      if (!child.name) continue;
      const childKind = child.kind ?? "";
      if (childKind === "function" || childKind === "method") {
        cls.methods.push({
          name: child.name,
          signature: buildKotlinSignature(child),
          docstring: child.description?.trim() || void 0,
          parameters: child.parameters?.map((p) => ({
            name: p.name,
            type: p.type ?? void 0,
            description: p.description?.trim() || void 0,
            default: p.defaultValue ?? void 0
          })),
          return_type: child.returnType ?? void 0
        });
      } else if (childKind === "property" || childKind === "field") {
        cls.properties.push({
          name: child.name,
          type: child.returnType ?? child.parameters?.[0]?.type ?? void 0,
          docstring: child.description?.trim() || void 0
        });
      }
    }
    mod.classes?.push(cls);
  } else if (kind === "function") {
    mod.functions?.push({
      name: node.name,
      signature: buildKotlinSignature(node),
      docstring: node.description?.trim() || void 0,
      parameters: node.parameters?.map((p) => ({
        name: p.name,
        type: p.type ?? void 0,
        description: p.description?.trim() || void 0,
        default: p.defaultValue ?? void 0
      })),
      return_type: node.returnType ?? void 0
    });
  } else if (kind === "property" || kind === "field") {
    mod.variables?.push({
      name: node.name,
      type: node.returnType ?? void 0,
      docstring: node.description?.trim() || void 0
    });
  }
  for (const child of node.children ?? []) {
    convertDokkaNode(child, mod);
  }
}
function buildKotlinSignature(node) {
  const params = (node.parameters ?? []).map((p) => {
    const defaultStr = p.defaultValue ? ` = ${p.defaultValue}` : "";
    return `${p.name}: ${p.type}${defaultStr}`;
  }).join(", ");
  const receiver = node.receiver ? `${node.receiver.type}.` : "";
  const returnType = node.returnType ? `: ${node.returnType}` : "";
  const modifiers = (node.modifiers ?? []).join(" ");
  const prefix = modifiers ? `${modifiers} ` : "";
  return `${prefix}${receiver}fun ${node.name}(${params})${returnType}`;
}

// handlers/php.ts
import { execSync as execSync9 } from "child_process";
import { existsSync as existsSync9, readFileSync as readFileSync7, readdirSync as readdirSync6 } from "fs";
import path11 from "path";
var phpHandler = {
  name: "php",
  async generate(options) {
    const opts = options;
    const entryPoints = opts.entryPoints;
    if (!entryPoints || entryPoints.length === 0) {
      throw new Error("PHP handler requires at least one entryPoint");
    }
    const modules = extractWithPhpDoc(entryPoints);
    if (modules.length === 0) {
      throw new Error("phpDocumentor extraction produced no modules");
    }
    const output = transformToMDX(modules, {
      outputDir: opts.output,
      language: "php",
      ...opts.pagination !== void 0 ? { pagination: opts.pagination } : {}
    });
    return output;
  },
  async validate(_sourcePath) {
    try {
      execSync9("phpdoc --version", { encoding: "utf-8", stdio: "pipe" });
      return { valid: true, errors: [] };
    } catch {
      return {
        valid: false,
        errors: [
          "phpDocumentor not found. Install with: composer global require phpdocumentor/phpdocumentor"
        ]
      };
    }
  }
};
function extractWithPhpDoc(entryPoints) {
  const modules = [];
  const tmpOutput = "/tmp/starlight-polyglot-phpdoc";
  const entries = entryPoints.map((e) => `"${path11.resolve(e)}"`).join(" ");
  const cmd = `phpdoc -t "${tmpOutput}" --template="xml" -f ${entries} -d "" 2>&1`;
  execSync9(cmd, {
    encoding: "utf-8",
    maxBuffer: 10 * 1024 * 1024,
    timeout: 18e4,
    stdio: "pipe"
  });
  if (existsSync9(tmpOutput)) {
    const xmlFiles = readdirSync6(tmpOutput).filter((f) => f.endsWith(".xml")).map((f) => path11.join(tmpOutput, f));
    for (const xmlFile of xmlFiles) {
      const raw = readFileSync7(xmlFile, "utf-8");
      const elements = parsePhpDocXml(raw);
      const converted = convertPhpDocElements(elements);
      modules.push(...converted);
    }
  }
  return modules;
}
function parsePhpDocXml(xmlContent) {
  const elements = [];
  const fileRegex = /<file[^>]*path="([^"]*)"[^>]*>([\s\S]*?)<\/file>/g;
  let fileMatch;
  while ((fileMatch = fileRegex.exec(xmlContent)) !== null) {
    const fileBody = fileMatch[2];
    const classRegex = /<class[^>]*>([\s\S]*?)<\/class>/g;
    let classMatch;
    while ((classMatch = classRegex.exec(fileBody)) !== null) {
      const classBody = classMatch[1];
      const className = classBody.match(/<full_name>([^<]*)<\/full_name>/)?.[1]?.trim();
      const classSummary = classBody.match(/<summary>([\s\S]*?)<\/summary>/)?.[1]?.trim();
      const classDesc = classBody.match(/<description>([\s\S]*?)<\/description>/)?.[1]?.trim();
      const element = {
        name: className,
        type: "class",
        summary: classSummary,
        description: classDesc,
        methods: [],
        properties: [],
        constants: []
      };
      const methodRegex = /<method[^>]*>([\s\S]*?)<\/method>/g;
      let methodMatch;
      while ((methodMatch = methodRegex.exec(classBody)) !== null) {
        const methodBody = methodMatch[1];
        const mName = methodBody.match(/<name>([^<]*)<\/name>/)?.[1]?.trim() ?? "unknown";
        const mSummary = methodBody.match(/<summary>([\s\S]*?)<\/summary>/)?.[1]?.trim();
        const mDesc = methodBody.match(/<description>([\s\S]*?)<\/description>/)?.[1]?.trim();
        const args = [];
        const argRegex = /<argument[^>]*>([\s\S]*?)<\/argument>/g;
        let argMatch;
        while ((argMatch = argRegex.exec(methodBody)) !== null) {
          const argBody = argMatch[1];
          const aName = argBody.match(/<name>([^<]*)<\/name>/)?.[1]?.trim() ?? "param";
          const aType = argBody.match(/<type>([^<]*)<\/type>/)?.[1]?.trim();
          const aDefault = argBody.match(/<default>([^<]*)<\/default>/)?.[1]?.trim();
          const aDesc = argBody.match(/<description>([^<]*)<\/description>/)?.[1]?.trim();
          args.push({ name: aName, type: aType || void 0, default: aDefault || void 0, description: aDesc || void 0 });
        }
        const returnMatch = methodBody.match(/<return[^>]*>([\s\S]*?)<\/return>/);
        let returnType;
        if (returnMatch) {
          returnType = returnMatch[1].match(/<type>([^<]*)<\/type>/)?.[1]?.trim() || void 0;
        }
        element.methods?.push({
          name: mName,
          type: "method",
          summary: mSummary,
          description: mDesc,
          arguments: args.length > 0 ? args : void 0,
          return: returnType ? { type: returnType } : void 0
        });
      }
      const propRegex = /<property[^>]*>([\s\S]*?)<\/property>/g;
      let propMatch;
      while ((propMatch = propRegex.exec(classBody)) !== null) {
        const propBody = propMatch[1];
        const pName = propBody.match(/<name>([^<]*)<\/name>/)?.[1]?.trim() ?? "unknown";
        const pType = propBody.match(/<type>([^<]*)<\/type>/)?.[1]?.trim();
        const pDefault = propBody.match(/<default>([^<]*)<\/default>/)?.[1]?.trim();
        const pSummary = propBody.match(/<summary>([^<]*)<\/summary>/)?.[1]?.trim();
        element.properties?.push({
          name: pName,
          type: pType || void 0,
          default: pDefault || void 0,
          summary: pSummary || void 0
        });
      }
      if (element.name) {
        elements.push(element);
      }
    }
    const funcRegex = /<function[^>]*>([\s\S]*?)<\/function>/g;
    let funcMatch;
    while ((funcMatch = funcRegex.exec(fileBody)) !== null) {
      const funcBody = funcMatch[1];
      const fName = funcBody.match(/<name>([^<]*)<\/name>/)?.[1]?.trim() ?? "unknown";
      const fSummary = funcBody.match(/<summary>([^<]*)<\/summary>/)?.[1]?.trim();
      const fDesc = funcBody.match(/<description>([^<]*)<\/description>/)?.[1]?.trim();
      const args = [];
      const argRegex = /<argument[^>]*>([\s\S]*?)<\/argument>/g;
      let argMatch;
      while ((argMatch = argRegex.exec(funcBody)) !== null) {
        const argBody = argMatch[1];
        const aName = argBody.match(/<name>([^<]*)<\/name>/)?.[1]?.trim() ?? "param";
        const aType = argBody.match(/<type>([^<]*)<\/type>/)?.[1]?.trim();
        const aDefault = argBody.match(/<default>([^<]*)<\/default>/)?.[1]?.trim();
        args.push({ name: aName, type: aType || void 0, default: aDefault || void 0 });
      }
      const returnMatch = funcBody.match(/<return[^>]*>([\s\S]*?)<\/return>/);
      let returnType;
      if (returnMatch) {
        returnType = returnMatch[1].match(/<type>([^<]*)<\/type>/)?.[1]?.trim() || void 0;
      }
      elements.push({
        name: fName,
        type: "function",
        summary: fSummary,
        description: fDesc,
        arguments: args.length > 0 ? args : void 0,
        return: returnType ? { type: returnType } : void 0
      });
    }
  }
  return elements;
}
function convertPhpDocElements(elements) {
  const moduleMap = /* @__PURE__ */ new Map();
  for (const el of elements) {
    const namespace = el.namespace ?? "Global";
    const modName = namespace.split("\\").pop() ?? namespace;
    if (!moduleMap.has(modName)) {
      moduleMap.set(modName, {
        name: modName,
        docstring: void 0,
        classes: [],
        functions: [],
        variables: []
      });
    }
    const mod = moduleMap.get(modName);
    if (el.type === "class" || el.type === "interface" || el.type === "trait") {
      const cls = {
        name: el.name ?? "Unknown",
        docstring: el.summary || el.description || void 0,
        methods: [],
        properties: []
      };
      for (const method of el.methods ?? []) {
        cls.methods.push({
          name: method.name ?? "unknown",
          signature: buildPhpSignature(method),
          docstring: method.summary || method.description || void 0,
          parameters: (method.arguments ?? []).map((a) => ({
            name: a.name,
            type: a.type ?? void 0,
            description: a.description ?? void 0,
            default: a.default ?? void 0
          })),
          return_type: method.return?.type ?? void 0
        });
      }
      for (const prop of el.properties ?? []) {
        cls.properties.push({
          name: prop.name,
          type: prop.type ?? void 0,
          docstring: prop.summary || prop.description || void 0
        });
      }
      mod.classes?.push(cls);
      if (!mod.docstring && cls.docstring) {
        mod.docstring = cls.docstring;
      }
    } else if (el.type === "function") {
      mod.functions?.push({
        name: el.name ?? "unknown",
        signature: buildPhpSignature(el),
        docstring: el.summary || el.description || void 0,
        parameters: (el.arguments ?? []).map((a) => ({
          name: a.name,
          type: a.type ?? void 0,
          description: a.description ?? void 0,
          default: a.default ?? void 0
        })),
        return_type: el.return?.type ?? void 0
      });
    }
  }
  return Array.from(moduleMap.values());
}
function buildPhpSignature(el) {
  const params = (el.arguments ?? []).map((a) => {
    const typeStr = a.type ? `${a.type} ` : "";
    const defaultStr = a.default !== void 0 ? ` = ${a.default}` : "";
    return `${typeStr}$${a.name}${defaultStr}`;
  }).join(", ");
  const returnType = el.return?.type ? `: ${el.return.type}` : "";
  return `function ${el.name}(${params})${returnType}`;
}

// handlers/python.ts
import { execSync as execSync10 } from "child_process";
import { existsSync as existsSync10 } from "fs";
import path12 from "path";
var pythonHandler = {
  name: "python",
  async generate(options) {
    const opts = options;
    const entryPoints = opts.entryPoints;
    if (!entryPoints || entryPoints.length === 0) {
      throw new Error("Python handler requires at least one entryPoint");
    }
    const ast = await extractWithGriffe(entryPoints, opts.pythonExecutable);
    const modules = ast.modules ?? [];
    if (modules.length === 0 && ast.errors) {
      throw new Error(`Python extraction failed: ${ast.errors.map((e) => e.error).join(", ")}`);
    }
    const output = transformToMDX(modules, {
      outputDir: opts.output,
      language: "python",
      ...opts.pagination !== void 0 ? { pagination: opts.pagination } : {}
    });
    return output;
  },
  async validate(sourcePath) {
    try {
      const result = execSync10('python3 -c "import griffe"', { encoding: "utf-8", stdio: "pipe" });
      return { valid: true, errors: [] };
    } catch {
      return { valid: false, errors: ["griffe not installed. Run: pip install griffe"] };
    }
  }
};
function extractWithGriffe(entryPoints, pythonExecutable = process.env.STARLIGHT_POLYGLOT_PYTHON ?? "python3") {
  const scriptPath = path12.resolve(__dirname, "..", "scripts", "python_extract.py");
  if (!existsSync10(scriptPath)) {
    throw new Error(`Python extraction script not found at ${scriptPath}`);
  }
  const args = [pythonExecutable, scriptPath, "--entry-points", ...entryPoints];
  const result = execSync10(args.join(" "), {
    encoding: "utf-8",
    maxBuffer: 10 * 1024 * 1024,
    // 10MB
    timeout: 6e4
  });
  return JSON.parse(result);
}

// handlers/r.ts
import { execSync as execSync11 } from "child_process";
import { existsSync as existsSync11 } from "fs";
import path13 from "path";
var rHandler = {
  name: "r",
  async generate(options) {
    const opts = options;
    const entryPoints = opts.entryPoints;
    if (!entryPoints || entryPoints.length === 0) {
      throw new Error("R handler requires at least one entryPoint");
    }
    const ast = await extractWithRScript(entryPoints);
    const modules = ast.modules ?? [];
    if (modules.length === 0 && ast.errors) {
      throw new Error(
        `R extraction failed: ${ast.errors.map((e) => e.error).join(", ")}`
      );
    }
    const output = transformToMDX(modules, {
      outputDir: opts.output,
      language: "r",
      ...opts.pagination !== void 0 ? { pagination: opts.pagination } : {}
    });
    return output;
  },
  async validate(_sourcePath) {
    try {
      execSync11("Rscript --version", { encoding: "utf-8", stdio: "pipe" });
      return { valid: true, errors: [] };
    } catch {
      return { valid: false, errors: ["Rscript not found. Install R from https://www.r-project.org/"] };
    }
  }
};
function extractWithRScript(entryPoints) {
  const scriptPath = path13.resolve(__dirname, "..", "scripts", "r_extract.R");
  if (!existsSync11(scriptPath)) {
    throw new Error(`R extraction script not found at ${scriptPath}`);
  }
  const args = ["Rscript", scriptPath, ...entryPoints];
  const result = execSync11(args.join(" "), {
    encoding: "utf-8",
    maxBuffer: 10 * 1024 * 1024,
    // 10MB
    timeout: 12e4
  });
  return JSON.parse(result);
}

// handlers/ruby.ts
import { execSync as execSync12 } from "child_process";
import { existsSync as existsSync12, readFileSync as readFileSync8 } from "fs";
import path14 from "path";
var rubyHandler = {
  name: "ruby",
  async generate(options) {
    const opts = options;
    const entryPoints = opts.entryPoints;
    if (!entryPoints || entryPoints.length === 0) {
      throw new Error("Ruby handler requires at least one entryPoint");
    }
    const modules = extractWithYard(entryPoints);
    if (modules.length === 0) {
      throw new Error("YARD extraction produced no modules");
    }
    const output = transformToMDX(modules, {
      outputDir: opts.output,
      language: "ruby",
      ...opts.pagination !== void 0 ? { pagination: opts.pagination } : {}
    });
    return output;
  },
  async validate(_sourcePath) {
    try {
      execSync12("yard --version", { encoding: "utf-8", stdio: "pipe" });
      return { valid: true, errors: [] };
    } catch {
      return {
        valid: false,
        errors: [
          "YARD not found. Install with: gem install yard"
        ]
      };
    }
  }
};
function extractWithYard(entryPoints) {
  const modules = [];
  for (const entry of entryPoints) {
    const resolvedEntry = path14.resolve(entry);
    if (!existsSync12(resolvedEntry)) {
      continue;
    }
    const tmpOutput = "/tmp/starlight-polyglot-yard";
    const cmd = `yard doc --output-format json --output "${tmpOutput}" "${resolvedEntry}" 2>&1`;
    execSync12(cmd, {
      encoding: "utf-8",
      cwd: existsSync12(resolvedEntry) && resolvedEntry.endsWith(".rb") ? path14.dirname(resolvedEntry) : resolvedEntry,
      maxBuffer: 10 * 1024 * 1024,
      timeout: 18e4,
      stdio: "pipe"
    });
    const yardJsonPath = path14.join(tmpOutput, "yard.json");
    if (existsSync12(yardJsonPath)) {
      const raw = readFileSync8(yardJsonPath, "utf-8");
      try {
        const yardData = JSON.parse(raw);
        const converted = convertYARDObjects(yardData);
        modules.push(...converted);
      } catch {
        continue;
      }
    }
  }
  return modules;
  function convertYARDObjects(yardObjects) {
    const modules2 = [];
    const moduleGroups = /* @__PURE__ */ new Map();
    for (const obj of yardObjects) {
      if (!obj.name) continue;
      const namespace = obj.namespace ?? "Global";
      if (!moduleGroups.has(namespace)) {
        moduleGroups.set(namespace, []);
      }
      moduleGroups.get(namespace).push(obj);
    }
    for (const [namespace, objs] of moduleGroups) {
      const mod = {
        name: namespace.split("::").pop() ?? namespace,
        docstring: void 0,
        classes: [],
        functions: [],
        variables: []
      };
      for (const obj of objs) {
        if (!obj.path) continue;
        if (obj.kind === "class" || obj.kind === "module") {
          const cls = {
            name: obj.name ?? obj.path.split("::").pop() ?? "Unknown",
            docstring: obj.docstring?.trim() || void 0,
            methods: [],
            properties: []
          };
          for (const child of obj.children ?? []) {
            if (child.kind === "method" || child.kind === "instance_method" || child.kind === "class_method") {
              cls.methods.push({
                name: child.name ?? "unknown",
                signature: buildRubySignature(child),
                docstring: child.docstring?.trim() || void 0,
                parameters: (child.params ?? []).map((p) => ({
                  name: p.name ?? "param",
                  type: p.types?.[0],
                  description: p.docstring?.trim() || void 0,
                  default: p.default ?? void 0
                })),
                return_type: child.return_types?.[0] ?? child.return_type ?? void 0
              });
            } else if (child.kind === "attribute" || child.kind === "attr_accessor" || child.kind === "attr_reader" || child.kind === "attr_writer") {
              cls.properties.push({
                name: child.name ?? "unknown",
                type: child.return_types?.[0] ?? void 0,
                docstring: child.docstring?.trim() || void 0
              });
            }
          }
          mod.classes?.push(cls);
          if (!mod.docstring && cls.docstring) {
            mod.docstring = cls.docstring;
          }
        } else if (obj.kind === "method" || obj.kind === "instance_method" || obj.kind === "class_method") {
          mod.functions?.push({
            name: obj.name ?? obj.path ?? "unknown",
            signature: buildRubySignature(obj),
            docstring: obj.docstring?.trim() || void 0,
            parameters: (obj.params ?? []).map((p) => ({
              name: p.name ?? "param",
              type: p.types?.[0],
              description: p.docstring?.trim() || void 0,
              default: p.default ?? void 0
            })),
            return_type: obj.return_types?.[0] ?? obj.return_type ?? void 0
          });
        } else if (obj.kind === "constant" || obj.kind === "variable" || obj.kind === "attr") {
          mod.variables?.push({
            name: obj.name ?? obj.path ?? "unknown",
            type: obj.return_types?.[0] ?? void 0,
            docstring: obj.docstring?.trim() || void 0
          });
        }
      }
      modules2.push(mod);
    }
    return modules2;
  }
  function buildRubySignature(obj) {
    const params = (obj.params ?? []).map((p) => {
      const base = p.name ?? "";
      if (p.default) return `${base} = ${p.default}`;
      return base;
    }).join(", ");
    const isClassMethod = obj.kind === "class_method";
    const prefix = isClassMethod ? "self." : "";
    return `${prefix}${obj.name}(${params})`;
  }
}

// handlers/rust.ts
import { execSync as execSync13 } from "child_process";
import { existsSync as existsSync13, readFileSync as readFileSync9 } from "fs";
import path15 from "path";
var rustHandler = {
  name: "rust",
  async generate(options) {
    const opts = options;
    const cratePath = opts.cratePath;
    if (!cratePath) {
      throw new Error("Rust handler requires a cratePath option");
    }
    if (!existsSync13(cratePath)) {
      throw new Error(`Crate path does not exist: ${cratePath}`);
    }
    const modules = extractWithRustDoc(cratePath);
    if (modules.length === 0) {
      throw new Error("rustdoc extraction produced no modules");
    }
    const output = transformToMDX(modules, {
      outputDir: opts.output,
      language: "rust",
      ...opts.pagination !== void 0 ? { pagination: opts.pagination } : {}
    });
    return output;
  },
  async validate(_sourcePath) {
    try {
      execSync13("cargo --version", { encoding: "utf-8", stdio: "pipe" });
      return { valid: true, errors: [] };
    } catch {
      return { valid: false, errors: ["cargo not found. Install Rust from https://rustup.rs/"] };
    }
  }
};
function buildRustSignature(item) {
  if (!item.decl) return void 0;
  const params = (item.decl.params ?? []).map((p) => `${p.name}: ${p.type}`).join(", ");
  const output = item.decl.output?.name ?? "";
  const header = item.header ?? {};
  let prefix = "";
  if (header.asyncness === "async") prefix += "async ";
  if (header.safety === "unsafe") prefix += "unsafe ";
  if (output && output !== "()" && output !== "unit") {
    return `${prefix}fn ${item.name}(${params}) -> ${output}`;
  }
  return `${prefix}fn ${item.name}(${params})`;
}
function extractRustParameters(item) {
  if (!item.decl?.params || item.decl.params.length === 0) return void 0;
  return item.decl.params.map((p) => ({
    name: p.name,
    type: p.type,
    description: void 0,
    default: void 0
  }));
}
function extractRustItem(item, _index) {
  if (item.kind !== "module" && item.kind !== "crate") return null;
  const mod = {
    name: item.name ?? "unknown",
    docstring: item.docs?.trim() || void 0,
    classes: [],
    functions: [],
    variables: []
  };
  for (const child of item.inner ?? []) {
    if (child.kind === "struct" || child.kind === "enum" || child.kind === "union" || child.kind === "trait") {
      const clsItem = {
        name: child.name ?? "unknown",
        docstring: child.docs?.trim() || void 0,
        methods: [],
        properties: []
      };
      for (const field of child.inner ?? []) {
        if (field.kind === "field") {
          clsItem.properties?.push({
            name: field.name ?? "unknown",
            type: field.decl?.output?.name ?? void 0,
            docstring: field.docs?.trim() || void 0
          });
        } else if (field.kind === "method") {
          clsItem.methods?.push({
            name: field.name ?? "unknown",
            signature: buildRustSignature(field),
            docstring: field.docs?.trim() || void 0,
            parameters: extractRustParameters(field),
            return_type: field.decl?.output?.name ?? void 0
          });
        }
      }
      mod.classes?.push(clsItem);
    } else if (child.kind === "function") {
      mod.functions?.push({
        name: child.name ?? "unknown",
        signature: buildRustSignature(child),
        docstring: child.docs?.trim() || void 0,
        parameters: extractRustParameters(child),
        return_type: child.decl?.output?.name ?? void 0
      });
    } else if (child.kind === "constant" || child.kind === "static") {
      mod.variables?.push({
        name: child.name ?? "unknown",
        type: child.decl?.output?.name ?? void 0,
        docstring: child.docs?.trim() || void 0
      });
    }
  }
  return mod;
}
function extractWithRustDoc(cratePath) {
  const resolvedPath = path15.resolve(cratePath);
  const cmd = `cargo +nightly rustdoc --output-format json --manifest-path "${resolvedPath}/Cargo.toml" 2>/dev/null`;
  execSync13(cmd, {
    encoding: "utf-8",
    stdio: "pipe",
    timeout: 12e4
  });
  const crateName = path15.basename(resolvedPath);
  const possiblePaths = [
    path15.resolve(resolvedPath, "target", "doc", `${crateName}.json`),
    path15.resolve(resolvedPath, "..", "target", "doc", `${crateName}.json`)
  ];
  let jsonPath = "";
  for (const p of possiblePaths) {
    if (existsSync13(p)) {
      jsonPath = p;
      break;
    }
  }
  if (!jsonPath) {
    throw new Error(
      `Could not find rustdoc JSON output. Expected at target/doc/${crateName}.json in crate or parent.`
    );
  }
  const raw = readFileSync9(jsonPath, "utf-8");
  const output = JSON.parse(raw);
  const modules = [];
  for (const item of Object.values(output.index)) {
    if (item.kind === "module" || item.kind === "crate") {
      const mod = extractRustItem(item, output.index);
      if (mod && !modules.find((m) => m.name === mod.name)) {
        modules.push(mod);
      }
    }
  }
  return modules;
}

// handlers/sas.ts
import { execSync as execSync14 } from "child_process";
import { existsSync as existsSync14 } from "fs";
import path16 from "path";
var sasHandler = {
  name: "sas",
  async generate(options) {
    const opts = options;
    const entryPoints = opts.entryPoints;
    if (!entryPoints || entryPoints.length === 0) {
      throw new Error("SAS handler requires at least one entryPoint");
    }
    const modules = extractWithSas(entryPoints);
    if (modules.length === 0) {
      throw new Error("SAS extraction produced no modules");
    }
    const output = transformToMDX(modules, {
      outputDir: opts.output,
      language: "sas",
      ...opts.pagination !== void 0 ? { pagination: opts.pagination } : {}
    });
    return output;
  },
  async validate(_sourcePath) {
    try {
      execSync14("sas -version", { encoding: "utf-8", stdio: "pipe" });
      return { valid: true, errors: [] };
    } catch {
      try {
        execSync14("sas --version", { encoding: "utf-8", stdio: "pipe" });
        return { valid: true, errors: [] };
      } catch {
        return {
          valid: false,
          errors: [
            "SAS not found. Install SAS from https://www.sas.com/ and ensure it is on your PATH."
          ]
        };
      }
    }
  }
};
function extractWithSas(entryPoints) {
  const modules = [];
  const scriptPath = path16.resolve(__dirname, "..", "scripts", "sas_extract.sas");
  if (!existsSync14(scriptPath)) {
    throw new Error(`SAS extraction script not found at ${scriptPath}`);
  }
  for (const entry of entryPoints) {
    const resolvedEntry = path16.resolve(entry);
    if (!existsSync14(resolvedEntry)) {
      continue;
    }
    const cmd = `sas -sysin "${scriptPath}" -set SRC_FILE "${resolvedEntry}" -log /tmp/sas_extract.log -print /tmp/sas_extract.lst 2>&1`;
    const result = execSync14(cmd, {
      encoding: "utf-8",
      cwd: path16.dirname(scriptPath),
      maxBuffer: 10 * 1024 * 1024,
      timeout: 18e4,
      stdio: "pipe"
    });
    const mod = parseSasOutput(result, resolvedEntry);
    if (mod) {
      modules.push(mod);
    }
  }
  return modules;
}
function parseSasOutput(output, entryPath) {
  const name = path16.basename(entryPath, path16.extname(entryPath));
  const lines = output.split("\n");
  const mod = {
    name,
    docstring: void 0,
    classes: [],
    functions: [],
    variables: []
  };
  const items = [];
  const macroRegex = /%\s*macro\s+(\w+)\s*\(([^)]*)\)\s*(?:\/\*\s*([^*]*)\s*\*\/)?/gi;
  let macroMatch;
  while ((macroMatch = macroRegex.exec(output)) !== null) {
    const macroName = macroMatch[1];
    const paramStr = macroMatch[2].trim();
    const description = macroMatch[3]?.trim();
    const params = paramStr ? paramStr.split(",").map((p) => {
      const parts = p.trim().split(/\s*=\s*/);
      return {
        name: parts[0]?.trim() || p.trim(),
        type: void 0,
        description: void 0,
        default: parts[1]?.trim() || void 0
      };
    }) : void 0;
    items.push({
      name: macroName,
      type: "macro",
      description: description || void 0,
      parameters: params && params.length > 0 ? params : void 0
    });
  }
  const varRegex = /^\s*(\d+)\s+(\w+)\s+(\w+)\s+(\d+)/gm;
  while ((macroMatch = varRegex.exec(output)) !== null) {
    const varName = macroMatch[2];
    const varType = macroMatch[3];
    items.push({
      name: varName,
      type: "variable",
      description: void 0,
      parameters: void 0,
      returns: varType
    });
  }
  const commentDocRegex = /\*\s*@(macro|function|dataset)\s+(\w+)\s*([^*]*)\*;/gi;
  while ((macroMatch = commentDocRegex.exec(output)) !== null) {
    const itemType = macroMatch[1];
    const itemName = macroMatch[2];
    const itemDesc = macroMatch[3]?.trim();
    const existing = items.find((i) => i.name === itemName);
    if (existing) {
      existing.description = itemDesc || existing.description;
    } else {
      items.push({
        name: itemName,
        type: itemType,
        description: itemDesc || void 0
      });
    }
  }
  for (const item of items) {
    if (item.type === "macro" || item.type === "function") {
      mod.functions?.push({
        name: item.name,
        signature: `${item.name}(${(item.parameters ?? []).map((p) => p.name).join(", ")})`,
        docstring: item.description,
        parameters: item.parameters?.map((p) => ({
          name: p.name,
          type: p.type,
          description: p.description,
          default: p.default
        })),
        return_type: item.returns
      });
    } else {
      mod.variables?.push({
        name: item.name,
        type: item.returns,
        docstring: item.description
      });
    }
  }
  if (!mod.docstring && mod.functions && mod.functions.length > 0) {
    mod.docstring = mod.functions[0].docstring;
  }
  return mod;
}

// handlers/scala.ts
import { execSync as execSync15 } from "child_process";
import { existsSync as existsSync15, readFileSync as readFileSync10, readdirSync as readdirSync7 } from "fs";
import path17 from "path";
var scalaHandler = {
  name: "scala",
  async generate(options) {
    const opts = options;
    const entryPoints = opts.entryPoints;
    if (!entryPoints || entryPoints.length === 0) {
      throw new Error("Scala handler requires at least one entryPoint");
    }
    const modules = extractWithScaladoc(entryPoints, opts.classpath);
    if (modules.length === 0) {
      throw new Error("Scaladoc extraction produced no modules");
    }
    const output = transformToMDX(modules, {
      outputDir: opts.output,
      language: "scala",
      ...opts.pagination !== void 0 ? { pagination: opts.pagination } : {}
    });
    return output;
  },
  async validate(_sourcePath) {
    try {
      execSync15("scaladoc --version", { encoding: "utf-8", stdio: "pipe" });
      return { valid: true, errors: [] };
    } catch {
      return {
        valid: false,
        errors: [
          "scaladoc not found. Install the Scala toolchain from https://www.scala-lang.org/download/"
        ]
      };
    }
  }
};
function extractWithScaladoc(entryPoints, classpath) {
  const tmpDir = "/tmp/starlight-polyglot-scaladoc";
  const entries = entryPoints.map((e) => `"${path17.resolve(e)}"`).join(" ");
  const cpFlag = classpath ? ` -classpath "${classpath}"` : "";
  const cmd = `scaladoc -d "${tmpDir}" -json${cpFlag} ${entries} 2>&1`;
  execSync15(cmd, {
    encoding: "utf-8",
    maxBuffer: 10 * 1024 * 1024,
    timeout: 18e4,
    stdio: "pipe"
  });
  if (!existsSync15(tmpDir)) {
    throw new Error(
      "scaladoc did not produce output. Ensure scaladoc is installed and entry points are valid."
    );
  }
  return parseScaladocDir(tmpDir);
}
function parseScaladocDir(dir) {
  const modules = [];
  const files = readdirSync7(dir).filter((f) => f.endsWith(".json"));
  for (const file of files) {
    const filePath = path17.join(dir, file);
    const raw = readFileSync10(filePath, "utf-8");
    try {
      const doc = JSON.parse(raw);
      const mod = convertScaladocDocument(doc);
      if (mod) {
        modules.push(mod);
      }
    } catch {
      continue;
    }
  }
  return modules;
  function convertScaladocDocument(doc) {
    if (!doc.name) return null;
    const mod = {
      name: doc.name,
      docstring: extractScaladocBody(doc.comment),
      classes: [],
      functions: [],
      variables: []
    };
    for (const member of doc.members ?? []) {
      const memberDoc = extractScaladocBody(member.comment);
      if (member.kind === "class" || member.kind === "trait" || member.kind === "object" || member.kind === "case class") {
        const cls = {
          name: member.name,
          docstring: memberDoc,
          methods: [],
          properties: []
        };
        for (const sub of member.members ?? []) {
          const subDoc = extractScaladocBody(sub.comment);
          if (sub.kind === "def" || sub.kind === "method" || sub.kind === "function") {
            cls.methods.push({
              name: sub.name,
              signature: buildScalaSignature(sub),
              docstring: subDoc,
              parameters: sub.params?.map((p) => ({
                name: p.name,
                type: p.typeName ?? void 0,
                description: extractParamDescription(sub.comment, p.name),
                default: p.defaultValue ?? void 0
              })),
              return_type: sub.resultType ?? void 0
            });
          } else if (sub.kind === "val" || sub.kind === "var" || sub.kind === "lazy val") {
            cls.properties.push({
              name: sub.name,
              type: sub.resultType ?? sub.valueType ?? void 0,
              docstring: subDoc
            });
          }
        }
        mod.classes?.push(cls);
      } else if (member.kind === "def" || member.kind === "function" || member.kind === "method") {
        mod.functions?.push({
          name: member.name,
          signature: buildScalaSignature(member),
          docstring: memberDoc,
          parameters: member.params?.map((p) => ({
            name: p.name,
            type: p.typeName ?? void 0,
            description: extractParamDescription(member.comment, p.name),
            default: p.defaultValue ?? void 0
          })),
          return_type: member.resultType ?? void 0
        });
      } else if (member.kind === "val" || member.kind === "var") {
        mod.variables?.push({
          name: member.name,
          type: member.resultType ?? member.valueType ?? void 0,
          docstring: memberDoc
        });
      }
    }
    return mod;
  }
  function extractScaladocBody(comment) {
    if (!comment) return void 0;
    const text = comment.body?.text ?? comment.body?.blocks?.map((b) => b.text).join("\n") ?? "";
    return text.trim() || void 0;
  }
  function extractParamDescription(comment, paramName) {
    if (!comment?.tags || !paramName) return void 0;
    const tag = comment.tags.find(
      (t) => t.tag === "@param" && t.paramName === paramName
    );
    return tag?.text?.trim() || void 0;
  }
  function buildScalaSignature(member) {
    const params = (member.params ?? []).map((p) => `${p.name}: ${p.typeName ?? "Any"}`).join(", ");
    const returnType = member.resultType ? `: ${member.resultType}` : "";
    return `${member.name}(${params})${returnType}`;
  }
}

// handlers/stata.ts
import { execSync as execSync16 } from "child_process";
import { existsSync as existsSync16 } from "fs";
import path18 from "path";
var stataHandler = {
  name: "stata",
  async generate(options) {
    const opts = options;
    const entryPoints = opts.entryPoints;
    if (!entryPoints || entryPoints.length === 0) {
      throw new Error("Stata handler requires at least one entryPoint");
    }
    const modules = extractWithStata(entryPoints);
    if (modules.length === 0) {
      throw new Error("Stata extraction produced no modules");
    }
    const output = transformToMDX(modules, {
      outputDir: opts.output,
      language: "stata",
      ...opts.pagination !== void 0 ? { pagination: opts.pagination } : {}
    });
    return output;
  },
  async validate(_sourcePath) {
    try {
      execSync16("stata --version", { encoding: "utf-8", stdio: "pipe" });
      return { valid: true, errors: [] };
    } catch {
      try {
        execSync16("stata-se --version", { encoding: "utf-8", stdio: "pipe" });
        return { valid: true, errors: [] };
      } catch {
        return {
          valid: false,
          errors: [
            "Stata not found. Install Stata from https://www.stata.com/ and ensure it is on your PATH."
          ]
        };
      }
    }
  }
};
function extractWithStata(entryPoints) {
  const modules = [];
  const scriptPath = path18.resolve(__dirname, "..", "scripts", "stata_extract.do");
  if (!existsSync16(scriptPath)) {
    throw new Error(`Stata extraction script not found at ${scriptPath}`);
  }
  for (const entry of entryPoints) {
    const resolvedEntry = path18.resolve(entry);
    if (!existsSync16(resolvedEntry)) {
      continue;
    }
    const cmd = `stata -b do "${scriptPath}" "${resolvedEntry}"`;
    const result = execSync16(cmd, {
      encoding: "utf-8",
      cwd: path18.dirname(scriptPath),
      maxBuffer: 10 * 1024 * 1024,
      timeout: 12e4,
      stdio: "pipe"
    });
    const mod = parseStataHelpOutput(result, resolvedEntry);
    if (mod) {
      modules.push(mod);
    }
  }
  return modules;
}
function parseStataHelpOutput(output, entryPath) {
  const name = path18.basename(entryPath, path18.extname(entryPath));
  const lines = output.split("\n").map((l) => l.trim()).filter(Boolean);
  if (lines.length === 0) return null;
  const mod = {
    name,
    docstring: void 0,
    classes: [],
    functions: [],
    variables: []
  };
  const syntaxIdx = lines.findIndex(
    (l) => /^syntax/i.test(l) || /^---+/i.test(l)
  );
  const descriptionEndIdx = syntaxIdx > 0 ? syntaxIdx : Math.min(lines.length, 5);
  const descriptionLines = lines.slice(0, descriptionEndIdx).filter(
    (l) => !/^(help|title)/i.test(l) && l.length > 0
  );
  if (descriptionLines.length > 0) {
    mod.docstring = descriptionLines.join(" ").replace(/\s+/g, " ").trim();
  }
  const optionLines = lines.filter(
    (l) => /^\s*[-–—]\s+\w/.test(l) || /^\s*\w+\s+\(/.test(l)
  );
  if (optionLines.length > 0) {
    mod.functions?.push({
      name,
      signature: `Syntax: ${lines[syntaxIdx] ?? name}`,
      docstring: mod.docstring,
      parameters: optionLines.map((line) => {
        const cleaned = line.replace(/^[\s\-–—]+/, "").trim();
        const colonIdx = cleaned.indexOf(":");
        const spaceIdx = cleaned.indexOf(" ");
        const splitIdx = colonIdx > 0 && (spaceIdx < 0 || colonIdx < spaceIdx) ? colonIdx : spaceIdx > 0 ? spaceIdx : cleaned.length;
        const paramName = cleaned.substring(0, splitIdx).trim();
        const paramDesc = cleaned.substring(splitIdx + 1).replace(/^[: ]+/, "").trim();
        return {
          name: paramName || "option",
          type: void 0,
          description: paramDesc || void 0,
          default: void 0
        };
      })
    });
  }
  return mod;
}

// handlers/swift.ts
import { execSync as execSync17 } from "child_process";
import { existsSync as existsSync17, readFileSync as readFileSync12, readdirSync as readdirSync8 } from "fs";
import path19 from "path";
var swiftHandler = {
  name: "swift",
  async generate(options) {
    const opts = options;
    const modulePath = opts.modulePath;
    const symbolGraphDir = opts.symbolGraphDir;
    if (!modulePath) {
      throw new Error("Swift handler requires a modulePath option");
    }
    if (!existsSync17(modulePath)) {
      throw new Error(`Module path does not exist: ${modulePath}`);
    }
    const modules = extractWithSwiftDoc(modulePath, symbolGraphDir);
    if (modules.length === 0) {
      throw new Error("Swift doc extraction produced no modules");
    }
    const output = transformToMDX(modules, {
      outputDir: opts.output,
      language: "swift",
      ...opts.pagination !== void 0 ? { pagination: opts.pagination } : {}
    });
    return output;
  },
  async validate(_sourcePath) {
    try {
      execSync17("swift-doc --version", { encoding: "utf-8", stdio: "pipe" });
      return { valid: true, errors: [] };
    } catch {
      try {
        execSync17("swift --version", { encoding: "utf-8", stdio: "pipe" });
        return { valid: true, errors: [] };
      } catch {
        return {
          valid: false,
          errors: [
            "Neither swift-doc nor swift toolchain found. Install swift-doc (https://github.com/SwiftDocOrg/swift-doc) or the Swift toolchain."
          ]
        };
      }
    }
  }
};
function extractWithSwiftDoc(modulePath, symbolGraphDir) {
  const resolvedPath = path19.resolve(modulePath);
  if (symbolGraphDir && existsSync17(symbolGraphDir)) {
    return parseSymbolGraphDir(symbolGraphDir);
  }
  const symbolGraphCandidates = findSymbolGraphFiles(resolvedPath);
  if (symbolGraphCandidates.length > 0) {
    return parseSymbolGraphDir(path19.dirname(symbolGraphCandidates[0]));
  }
  return runSwiftDoc(resolvedPath);
}
function findSymbolGraphFiles(modulePath) {
  const results = [];
  function searchDir(dir) {
    try {
      const entries = readdirSync8(dir, { withFileTypes: true });
      for (const entry of entries) {
        const fullPath = path19.join(dir, entry.name);
        if (entry.isDirectory() && !entry.name.startsWith(".") && entry.name !== "node_modules") {
          searchDir(fullPath);
        } else if (entry.isFile() && (entry.name.endsWith(".symbolgraph.json") || entry.name.endsWith(".json"))) {
          results.push(fullPath);
        }
      }
    } catch {
    }
  }
  const symbolGraphDir = path19.join(modulePath, ".build", "symbolgraph");
  if (existsSync17(symbolGraphDir)) {
    searchDir(symbolGraphDir);
  }
  searchDir(modulePath);
  return results;
}
function runSwiftDoc(modulePath) {
  const tmpOutput = path19.resolve(modulePath, ".build", "swift-doc-output");
  const cmd = `swift-doc --output "${tmpOutput}" --format json "${modulePath}" 2>&1`;
  execSync17(cmd, {
    encoding: "utf-8",
    cwd: modulePath,
    stdio: "pipe",
    timeout: 18e4
  });
  if (!existsSync17(tmpOutput)) {
    throw new Error(
      "swift-doc did not produce output. Ensure swift-doc is installed and the module path is correct."
    );
  }
  return parseSymbolGraphDir(tmpOutput);
}
function parseSymbolGraphDir(symbolGraphDir) {
  const modules = [];
  const files = readdirSync8(symbolGraphDir).filter(
    (f) => f.endsWith(".json") || f.endsWith(".symbolgraph.json")
  );
  for (const file of files) {
    const filePath = path19.join(symbolGraphDir, file);
    const raw = readFileSync12(filePath, "utf-8");
    try {
      const doc = JSON.parse(raw);
      const mod = convertSymbolGraph(doc);
      if (mod) {
        modules.push(mod);
      }
    } catch {
      continue;
    }
  }
  return modules;
}
function convertSymbolGraph(doc) {
  const moduleName = doc.module?.name ?? "UnknownModule";
  const mod = {
    name: moduleName,
    docstring: void 0,
    classes: [],
    functions: [],
    variables: []
  };
  if (!doc.symbols || doc.symbols.length === 0) return mod;
  const parentMap = /* @__PURE__ */ new Map();
  for (const rel of doc.relationships ?? []) {
    if (rel.kind === "memberOf" && rel.target) {
      parentMap.set(rel.source, rel.target);
    }
  }
  const childSymbols = /* @__PURE__ */ new Map();
  const topLevelSymbols = [];
  for (const symbol of doc.symbols) {
    const parent = parentMap.get(symbol.identifier?.precise ?? "");
    if (parent) {
      const children = childSymbols.get(parent) ?? [];
      children.push(symbol);
      childSymbols.set(parent, children);
    } else {
      topLevelSymbols.push(symbol);
    }
  }
  for (const symbol of topLevelSymbols) {
    const kind = symbol.kind?.identifier ?? "";
    const name = symbol.names?.title;
    if (!name) continue;
    const docComment = extractDocComment(symbol);
    if (kind === "class" || kind === "struct" || kind === "enum" || kind === "protocol" || kind === "extension") {
      const cls = {
        name,
        docstring: docComment,
        methods: [],
        properties: []
      };
      const preciseId = symbol.identifier?.precise ?? "";
      const children = childSymbols.get(preciseId) ?? [];
      for (const child of children) {
        const childKind = child.kind?.identifier ?? "";
        const childName = child.names?.title;
        if (!childName) continue;
        const childDoc = extractDocComment(child);
        if (childKind === "method" || childKind === "instanceMethod" || childKind === "typeMethod" || childKind === "constructor" || childKind === "instanceSubscript") {
          cls.methods.push({
            name: childName,
            signature: buildSwiftSignature(child),
            docstring: childDoc,
            parameters: extractSwiftParameters(child),
            return_type: extractSwiftReturnType(child)
          });
        } else if (childKind === "property" || childKind === "instanceProperty" || childKind === "typeProperty" || childKind === "instanceVariable") {
          cls.properties.push({
            name: childName,
            type: extractSwiftReturnType(child) ?? void 0,
            docstring: childDoc
          });
        }
      }
      mod.classes?.push(cls);
    } else if (kind === "function" || kind === "operator" || kind === "instanceMethod" || kind === "typeMethod") {
      mod.functions?.push({
        name,
        signature: buildSwiftSignature(symbol),
        docstring: docComment,
        parameters: extractSwiftParameters(symbol),
        return_type: extractSwiftReturnType(symbol)
      });
    } else if (kind === "variable" || kind === "global" || kind === "typealias") {
      mod.variables?.push({
        name,
        type: extractSwiftReturnType(symbol) ?? void 0,
        docstring: docComment
      });
    }
  }
  return mod;
}
function extractDocComment(symbol) {
  if (!symbol.docComment?.lines) return void 0;
  const text = symbol.docComment.lines.map((l) => l.text).join("").trim();
  return text || void 0;
}
function buildSwiftSignature(symbol) {
  const sig = symbol.functionSignature;
  if (!sig) return void 0;
  const params = (sig.parameters ?? []).map((p) => {
    const external = p.externalName ?? "_";
    const type = p.declarationMeta?.typeName ?? "Any";
    return `${external} ${p.name}: ${type}`;
  }).join(", ");
  const returns = sig.returns && sig.returns.length > 0 ? ` -> ${sig.returns.map((r) => r.name).join(", ")}` : "";
  return `${symbol.names.title}(${params})${returns}`;
}
function extractSwiftParameters(symbol) {
  const sig = symbol.functionSignature;
  if (!sig?.parameters || sig.parameters.length === 0) return void 0;
  return sig.parameters.map((p) => ({
    name: p.name,
    type: p.declarationMeta?.typeName ?? void 0,
    description: void 0,
    default: void 0
  }));
}
function extractSwiftReturnType(symbol) {
  const returns = symbol.functionSignature?.returns;
  if (!returns || returns.length === 0) return void 0;
  return returns.map((r) => r.name).join(", ") || void 0;
}

// handlers/typescript.ts
var typescriptHandler = {
  name: "typescript",
  async generate(options) {
    const opts = options;
    const entryPoints = opts.entryPoints;
    const tsconfig = opts.tsconfig;
    if (!entryPoints || entryPoints.length === 0) {
      throw new Error("TypeScript handler requires at least one entryPoint");
    }
    const modules = await extractWithTypeDoc(entryPoints, tsconfig);
    if (modules.length === 0) {
      throw new Error("TypeDoc extraction produced no modules");
    }
    const output = transformToMDX(modules, {
      outputDir: opts.output,
      language: "typescript",
      ...opts.pagination !== void 0 ? { pagination: opts.pagination } : {}
    });
    return output;
  },
  async validate(_sourcePath) {
    try {
      const typedoc = await import("typedoc");
      if (!typedoc.Application) {
        return { valid: false, errors: ["typedoc module loaded but Application class not found"] };
      }
      return { valid: true, errors: [] };
    } catch {
      return { valid: false, errors: ["typedoc not installed. Run: npm install typedoc typedoc-plugin-markdown"] };
    }
  }
};
function extractCommentText(comment) {
  if (!comment?.summary) return void 0;
  return comment.summary.map((part) => part.text).join("").trim() || void 0;
}
function extractSignature(reflection) {
  if (!reflection.signatures || reflection.signatures.length === 0) return void 0;
  const sig = reflection.signatures[0];
  const params = (sig.parameters ?? []).map((p) => {
    const typeName = p.type?.name ?? p.type?.type ?? "any";
    return `${p.name}: ${typeName}`;
  }).join(", ");
  const returnType = sig.type?.name ?? sig.type?.type ?? "void";
  return `${reflection.name}(${params}): ${returnType}`;
}
function extractReturnType(reflection) {
  if (!reflection.signatures || reflection.signatures.length === 0) return void 0;
  return reflection.signatures[0].type?.name ?? reflection.signatures[0].type?.type ?? void 0;
}
function extractParameters(reflection) {
  if (!reflection.signatures || reflection.signatures.length === 0) return void 0;
  const sig = reflection.signatures[0];
  if (!sig.parameters || sig.parameters.length === 0) return void 0;
  return sig.parameters.map((p) => ({
    name: p.name,
    type: p.type?.name ?? p.type?.type ?? void 0,
    description: extractCommentText(p.comment),
    default: p.defaultValue ?? void 0
  }));
}
function convertReflectionToASTModules(reflections) {
  const modules = [];
  for (const ref of reflections) {
    if (ref.kind === 1 || ref.kind === 2) {
      const mod = {
        name: ref.name,
        docstring: extractCommentText(ref.comment),
        classes: [],
        functions: [],
        variables: []
      };
      for (const child of ref.children ?? []) {
        if (child.kind === 128) {
          mod.classes?.push({
            name: child.name,
            docstring: extractCommentText(child.comment),
            methods: child.children?.filter((m) => m.kind === 256 || m.kind === 512).map((m) => ({
              name: m.name,
              signature: extractSignature(m),
              docstring: extractCommentText(m.comment),
              parameters: extractParameters(m),
              return_type: extractReturnType(m)
            })),
            properties: child.children?.filter((m) => m.kind === 1024).map((m) => ({
              name: m.name,
              type: m.type?.name ?? m.type?.type ?? void 0,
              docstring: extractCommentText(m.comment)
            }))
          });
        } else if (child.kind === 64) {
          mod.functions?.push({
            name: child.name,
            signature: extractSignature(child),
            docstring: extractCommentText(child.comment),
            parameters: extractParameters(child),
            return_type: extractReturnType(child)
          });
        } else if (child.kind === 1024) {
          mod.variables?.push({
            name: child.name,
            type: child.type?.name ?? child.type?.type ?? void 0,
            docstring: extractCommentText(child.comment)
          });
        }
      }
      modules.push(mod);
    }
  }
  return modules;
}
async function extractWithTypeDoc(entryPoints, tsconfig) {
  const { Application, TSConfigReader } = await import("typedoc");
  const app = await Application.bootstrap({
    entryPoints,
    tsconfig,
    skipErrorChecking: false,
    excludeExternals: true,
    excludePrivate: true,
    excludeProtected: false,
    validation: { notExported: false },
    plugin: []
  });
  app.options.addReader(new TSConfigReader());
  const project = await app.convert();
  if (!project) {
    throw new Error("TypeDoc conversion returned no project. Check entry points and tsconfig.");
  }
  const serialized = app.serializer.projectToObject(project);
  const children = serialized.children ?? [];
  return convertReflectionToASTModules(children);
}

// core/router.ts
function resolveHandlers(config, logger) {
  const handlers = [];
  const handlerMap = getHandlerMap();
  for (const [lang, opts] of Object.entries(config)) {
    if (!opts) continue;
    const language = lang;
    const handler = handlerMap[language];
    if (!handler) {
      logger.warn(`[starlight-polyglot] Unknown language "${lang}", skipping`);
      continue;
    }
    const output = opts.output ?? `api/${lang}`;
    const options = {
      ...opts,
      output
    };
    handlers.push({ name: language, handler, options });
  }
  if (handlers.length === 0) {
    logger.warn("[starlight-polyglot] No handlers configured. Add at least one language to your polyglot config.");
  }
  return handlers;
}
function getHandlerMap() {
  return {
    // Phase 1 handlers — registered at build time
    python: registeredHandler("python", pythonHandler),
    typescript: registeredHandler("typescript", typescriptHandler),
    rust: registeredHandler("rust", rustHandler),
    r: registeredHandler("r", rHandler),
    julia: registeredHandler("julia", juliaHandler),
    csharp: registeredHandler("csharp", csharpHandler),
    go: registeredHandler("go", goHandler),
    // Phase 2 handlers — Java ecosystem, C++, Swift
    java: registeredHandler("java", javaHandler),
    kotlin: registeredHandler("kotlin", kotlinHandler),
    cpp: registeredHandler("cpp", cppHandler),
    swift: registeredHandler("swift", swiftHandler),
    // Phase 3 handlers — Data science & scripting
    stata: registeredHandler("stata", stataHandler),
    sas: registeredHandler("sas", sasHandler),
    // Phase 4 handlers — JVM/CLR ecosystem
    scala: registeredHandler("scala", scalaHandler),
    // Phase 5 handlers — Dynamic & functional languages
    ruby: registeredHandler("ruby", rubyHandler),
    dart: registeredHandler("dart", dartHandler),
    php: registeredHandler("php", phpHandler),
    elixir: registeredHandler("elixir", elixirHandler)
  };
}
function registeredHandler(name, handler) {
  return {
    name,
    async generate(options) {
      return Promise.resolve().then(() => handler.generate(options));
    },
    ...handler.validate ? { validate: (sourcePath) => handler.validate(sourcePath) } : {}
  };
}

// index.ts
var sidebarGroup = getSidebarGroupPlaceholder();
function polyglot(options) {
  return makePolyglotPlugin(sidebarGroup)(options);
}
function createPolyglotPlugin() {
  const group = getSidebarGroupPlaceholder(Symbol(randomBytes(24).toString("base64url")));
  return [makePolyglotPlugin(group), group];
}
function makePolyglotPlugin(sidebarGroup2) {
  return function polyglotPlugin(options) {
    return {
      name: "starlight-polyglot",
      hooks: {
        async "config:setup"({ astroConfig, command, config, logger, updateConfig }) {
          if (command === "preview") return;
          const handlers = resolveHandlers(options, logger);
          const outputs = [];
          for (const handler of handlers) {
            try {
              logger.info(`[starlight-polyglot] Generating ${handler.name} documentation...`);
              const handlerOptions = handler.options;
              const output = await handler.handler.generate(handlerOptions);
              outputs.push(output);
              logger.info(`[starlight-polyglot] \u2713 ${handler.name}: ${output.pages.length} pages generated`);
            } catch (error) {
              logger.error(`[starlight-polyglot] \u2717 ${handler.name}: ${error.message}`);
              throw error;
            }
          }
          updateConfig({
            sidebar: mergeSidebars(config.sidebar, sidebarGroup2, outputs)
          });
        }
      }
    };
  };
}
function mergeSidebars(existingSidebar, group, outputs) {
  const sidebar = Array.isArray(existingSidebar) ? [...existingSidebar] : [];
  const apiGroups = outputs.filter((o) => o.sidebar).map((o) => o.sidebar);
  if (apiGroups.length > 0) {
    const placeholderIndex = sidebar.findIndex(
      (item) => typeof item === "object" && item !== null && item._key === group._key
    );
    if (placeholderIndex >= 0) {
      sidebar[placeholderIndex] = apiGroups.length === 1 ? apiGroups[0] : { label: "API", items: apiGroups };
    } else {
      sidebar.push(apiGroups.length === 1 ? apiGroups[0] : { label: "API", items: apiGroups });
    }
  }
  return sidebar;
}
export {
  createPolyglotPlugin,
  polyglot as default,
  sidebarGroup
};
