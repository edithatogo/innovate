import { afterEach, describe, expect, it, vi } from "vitest";

const spawnSyncMock = vi.hoisted(() => vi.fn());

vi.mock("node:child_process", () => ({
  spawnSync: spawnSyncMock,
}));

import { checkSchemaVersion } from "../scripts/check-schema-version";

describe("schema version check script", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    delete process.env.INNOVATE_PYTHON_COMMAND;
  });

  it("returns the Python schema version when both sides match", () => {
    spawnSyncMock.mockReturnValue({
      error: undefined,
      status: 0,
      stdout: "1.0\n",
      stderr: "",
    });

    expect(checkSchemaVersion()).toBe("1.0");
  });

  it("rejects schema drift between TypeScript and Python", () => {
    spawnSyncMock.mockReturnValue({
      error: undefined,
      status: 0,
      stdout: "2.0\n",
      stderr: "",
    });

    expect(() => checkSchemaVersion()).toThrow(/drift detected/);
  });

  it("rejects a failing Python command", () => {
    spawnSyncMock.mockReturnValue({
      error: undefined,
      status: 1,
      stdout: "",
      stderr: "boom",
    });

    expect(() => checkSchemaVersion()).toThrow(/Failed to read Python kernel schema version/);
  });

  it("rejects an empty launcher command", () => {
    process.env.INNOVATE_PYTHON_COMMAND = "   ";

    expect(() => checkSchemaVersion()).toThrow(/must not be empty/);
  });
});
