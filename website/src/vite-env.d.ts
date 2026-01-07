/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly PACKAGE_VERSION: string;
  readonly PACKAGE_BUILD_DATE: string;
  readonly GIT_COMMIT_SHA_SHORT: string;
  // FB build flag - true when built with FB_BUILD=true environment variable
  readonly IS_FB_BUILD: boolean;
  // Internal wiki URL - only set when IS_FB_BUILD is true
  readonly INTERNAL_WIKI_URL: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
