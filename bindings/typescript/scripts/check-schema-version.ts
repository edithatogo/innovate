import { spawnSync } from "node:child_process";
import { delimiter, join } from "node:path";
import { pathToFileURL } from "node:url";

import { KERNEL_SCHEMA_VERSION, kernelPythonCommand, kernelRepoRoot } from "../src";

export function checkSchemaVersion(): string {
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

  if (result.error) {
    throw result.error;
  }

  if (result.status !== 0) {
    throw new Error(
      `Failed to read Python kernel schema version: ${result.stderr || result.stdout || "unknown error"}`,
    );
  }

  const pythonVersion = result.stdout.trim();
  if (pythonVersion !== KERNEL_SCHEMA_VERSION) {
    throw new Error(
      `Kernel schema version drift detected: TypeScript=${KERNEL_SCHEMA_VERSION} Python=${pythonVersion}`,
    );
  }

  return pythonVersion;
}

function main(): void {
  console.log(`Kernel schema versions match: ${checkSchemaVersion()}`);
}

const entrypoint = process.argv[1];
if (entrypoint && import.meta.url === pathToFileURL(entrypoint).href) {
  main();
}
