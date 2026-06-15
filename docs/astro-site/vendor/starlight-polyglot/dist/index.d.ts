import { StarlightPlugin } from '@astrojs/starlight/types';

/**
 * starlight-polyglot — Handler Interface
 *
 * Defines the contract that every language handler must implement.
 * Handlers are responsible for extracting API documentation from
 * a language's source code and producing Starlight-compatible MDX output.
 *
 * @module core/handler
 */
/**
 * Supported programming languages.
 */
type Language = 'python' | 'typescript' | 'rust' | 'r' | 'julia' | 'csharp' | 'go' | 'java' | 'kotlin' | 'cpp' | 'swift' | 'stata' | 'sas' | 'scala' | 'ruby' | 'dart' | 'php' | 'elixir';
/**
 * A single MDX output page produced by a handler, used internally
 * when writing files to disk. This is the per-page representation.
 */
interface MDXOutput {
    /** Raw MDX content (frontmatter + body) */
    content: string;
    /** Parsed frontmatter metadata */
    frontmatter: Record<string, unknown>;
    /** Relative output path under the Starlight content directory, e.g. "api/python/io.mdx" */
    outputPath: string;
}
/**
 * A sidebar item linking to a generated page.
 */
interface SidebarItem$1 {
    label: string;
    link: string;
}
/**
 * A handler page with minimal frontmatter properties needed for
 * sidebar integration. The full MDX frontmatter is generated
 * during the MDX writing phase.
 */
interface HandlerPage$1 {
    path: string;
    frontmatter: Record<string, unknown>;
    body: string;
}
/**
 * Aggregate output from a handler, containing pages and sidebar.
 * This is what handlers actually return (from transformToMDX).
 * The pages array contains per-page data, and the sidebar
 * provides navigation structure for Starlight.
 */
interface HandlerAggregateOutput {
    pages: HandlerPage$1[];
    sidebar: {
        label: string;
        items: SidebarItem$1[];
    };
}
/**
 * Options passed to a handler's `generate()` method.
 * Each language handler may extend this with language-specific options.
 */
interface HandlerOptions {
    /** Output subdirectory under src/content/docs/, e.g. "api/python" */
    output: string;
    /** Whether to include pagination links between pages */
    pagination?: boolean;
    /** Whether to watch source files for changes */
    watch?: boolean;
    /** Arbitrary additional options forwarded from user configuration */
    [key: string]: unknown;
}
/**
 * Result of an optional handler validation step.
 */
interface ValidationResult {
    /** Whether the handler's environment/preconditions are satisfied */
    valid: boolean;
    /** Human-readable error messages describing what's missing */
    errors: string[];
}
/**
 * Handler contract.
 *
 * Every language handler MUST implement this interface to be
 * discoverable and executable by the starlight-polyglot plugin.
 */
interface Handler {
    /** Language identifier matching the Language union type */
    name: Language;
    /**
     * Generate MDX documentation pages from source code.
     *
     * @param options - Handler-specific configuration and output settings
     * @returns Aggregate output with pages array and sidebar configuration
     */
    generate(options: HandlerOptions): Promise<HandlerAggregateOutput>;
    /**
     * Optional pre-flight validation to check that the handler's
     * runtime environment (e.g., CLI tools, SDKs) is available.
     *
     * @param sourcePath - Path to the source code or project root
     * @returns Validation result indicating any issues
     */
    validate?(sourcePath: string): Promise<ValidationResult>;
}

/**
 * A symbol-based key used to identify the placeholder sidebar group.
 */
interface SidebarGroup {
    _key?: symbol;
    label: string;
    items: SidebarItem[];
}
interface SidebarItem {
    label?: string;
    link?: string;
    items?: SidebarItem[];
    autogenerate?: {
        directory: string;
    };
}
/**
 * Unified frontmatter schema for all generated MDX pages.
 */
interface MDXFrontmatter extends Record<string, unknown> {
    title: string;
    description?: string;
    sidebar: {
        label: string;
        order?: number;
    };
    pagefind?: boolean;
    /** The language this page documents */
    language?: string;
    /** The source module path */
    source?: string;
}
/**
 * Output from a single handler
 */
interface HandlerOutput extends HandlerAggregateOutput {
}
interface HandlerPage {
    /** Relative path within output directory, e.g. "python/io.mdx" */
    path: string;
    frontmatter: MDXFrontmatter;
    body: string;
}
/**
 * Standardized options passed to every handler.
 * Each language handler may extend this with language-specific options.
 *
 * @deprecated Use `HandlerOptions` from `./handler` instead.
 *   This alias exists for backward compatibility with existing handler implementations.
 */
interface BaseHandlerOptions extends HandlerOptions {
    /** Output subdirectory under src/content/docs/, e.g. "api/python" */
    output: string;
    /** Whether pagination links should be included */
    pagination?: boolean;
    /** Whether to watch for changes */
    watch?: boolean;
}

/**
 * Per-language handler configuration.
 */
interface HandlerConfig {
    entryPoints?: string[];
    output?: string;
    tsconfig?: string;
    cratePath?: string;
    modulePath?: string;
    projectPath?: string;
    pagination?: boolean;
    watch?: boolean;
    [key: string]: unknown;
}
/**
 * Overall plugin configuration.
 * Maps languages to their handler options.
 */
interface PolyglotConfig {
    python?: HandlerConfig;
    typescript?: HandlerConfig;
    rust?: HandlerConfig;
    r?: HandlerConfig;
    julia?: HandlerConfig;
    csharp?: HandlerConfig;
    go?: HandlerConfig;
    java?: HandlerConfig;
    kotlin?: HandlerConfig;
    cpp?: HandlerConfig;
    swift?: HandlerConfig;
    stata?: HandlerConfig;
    sas?: HandlerConfig;
    scala?: HandlerConfig;
    ruby?: HandlerConfig;
    dart?: HandlerConfig;
    php?: HandlerConfig;
    elixir?: HandlerConfig;
    [key: string]: HandlerConfig | undefined;
}

declare const sidebarGroup: SidebarGroup;
declare function polyglot(options: PolyglotConfig): StarlightPlugin;
declare function createPolyglotPlugin(): [plugin: typeof polyglot, group: SidebarGroup];

export { type BaseHandlerOptions, type Handler, type HandlerAggregateOutput, type MDXOutput as HandlerMDXOutput, type HandlerOptions, type HandlerOutput, type HandlerPage$1 as HandlerPage, type Language, type MDXFrontmatter, type HandlerPage as PluginHandlerPage, type PolyglotConfig, type SidebarGroup, type SidebarItem$1 as SidebarItem, type ValidationResult, createPolyglotPlugin, polyglot as default, sidebarGroup };
