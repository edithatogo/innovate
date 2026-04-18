import { spawnSync } from "node:child_process";
import { delimiter, join } from "node:path";

import { describe, expect, it } from "vitest";

import { KERNEL_SCHEMA_VERSION, kernelPythonCommand, kernelRepoRoot } from "../src";
import { checkSchemaVersion } from "../scripts/check-schema-version";

describe("Kernel schema compatibility", () => {
  it("keeps the TypeScript and Python kernel schema versions aligned", () => {
    expect(checkSchemaVersion()).toBe(KERNEL_SCHEMA_VERSION);

    const command = kernelPythonCommand().trim().split(/\s+/).filter(Boolean);
    if (command.length === 0) {
      throw new Error("INNOVATE_PYTHON_COMMAND must not be empty");
    }

    const result = spawnSync(
      command[0],
      [
        ...command.slice(1),
        "-c",
        "from innovate.kernel import KERNEL_SCHEMA_VERSION; print(KERNEL_SCHEMA_VERSION)",
      ],
      {
        cwd: kernelRepoRoot(),
        encoding: "utf8",
        env: {
          ...process.env,
          PYTHONPATH: [join(kernelRepoRoot(), "src"), process.env.PYTHONPATH]
            .filter((part) => Boolean(part))
            .join(delimiter),
        },
      },
    );

    expect(result.error).toBeUndefined();
    expect(result.status).toBe(0);
    expect(result.stdout.trim()).toBe(KERNEL_SCHEMA_VERSION);
  });
});
