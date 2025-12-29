/**
 * Global type declarations for TritonParse website
 */

declare global {
  interface Window {
    /** Temporary storage for left kernel hash during URL parsing */
    __TRITONPARSE_leftHash?: string;
    /** Temporary storage for right kernel hash during URL parsing */
    __TRITONPARSE_rightHash?: string;
  }
}

export {};
