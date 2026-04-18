import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

describe("TypeScript package scaffold", () => {
  it("includes a package README with installation and bridge guidance", () => {
    const readme = readFileSync(join(process.cwd(), "README.md"), "utf8");
    expect(readme).toContain("kernel_bridge.py");
    expect(readme).toContain("npm install");
  });
});
