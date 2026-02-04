import { useEffect, useCallback, useRef } from 'react';

/**
 * Message types for iframe communication protocol
 */
export type IframeMessageType = 'PING' | 'PONG' | 'DATA' | 'DATA_ACK' | 'ERROR';

/**
 * Options for the iframe messaging hook
 */
export interface UseIframeMessagingOptions {
  /** Callback when ArrayBuffer data is received */
  onDataReceived: (data: ArrayBuffer) => Promise<void>;
  /** Optional allowed origins for security (default: '*' allows all) */
  allowedOrigins?: string[];
  /** Enable debug logging */
  debug?: boolean;
}

/**
 * Custom hook for handling iframe postMessage communication
 * Implements PING/PONG protocol for connection health checks
 * and handles ArrayBuffer data transfer from parent window
 */
export function useIframeMessaging(options: UseIframeMessagingOptions) {
  const { onDataReceived, allowedOrigins = ['*'], debug = false } = options;
  const isReadyRef = useRef(false);
  const hasReceivedDataRef = useRef(false);

  const log = useCallback((...args: unknown[]) => {
    if (debug) {
      console.log('[IframeMessaging]', ...args);
    }
  }, [debug]);

  /**
   * Send a message to the parent window (simple string format)
   */
  const sendMessage = useCallback((message: string) => {
    if (window.parent && window.parent !== window) {
      log('Sending message:', message);
      window.parent.postMessage(message, '*');
    }
  }, [log]);

  /**
   * Handle incoming messages from parent window
   * Supports both simple string messages ('PING') and object messages with payload
   * Stops listening after first DATA message is processed
   */
  const handleMessage = useCallback(async (event: MessageEvent) => {
    // Security: check origin if specific origins are configured
    if (allowedOrigins[0] !== '*' && !allowedOrigins.includes(event.origin)) {
      log('Rejected message from unauthorized origin:', event.origin);
      return;
    }

    const data = event.data;

    // Handle simple string message: 'PING'
    if (data === 'PING') {
      log('Received PING');
      sendMessage('PONG');
      return;
    }

    // Handle object message with type field
    if (data && typeof data === 'object' && data.type) {
      log('Received message:', data.type);

      switch (data.type) {
        case 'PING':
          sendMessage('PONG');
          break;

        case 'DATA':
          // Ignore if we've already processed data (one-shot mode)
          if (hasReceivedDataRef.current) {
            log('Ignoring DATA message - already processed');
            return;
          }

          // Handle ArrayBuffer data
          try {
            let arrayBuffer: ArrayBuffer;

            if (data.payload instanceof ArrayBuffer) {
              arrayBuffer = data.payload;
            } else if (
              data.payload &&
              typeof data.payload === 'object' &&
              'byteLength' in data.payload
            ) {
              // Handle case where ArrayBuffer was serialized
              arrayBuffer = data.payload as ArrayBuffer;
            } else {
              throw new Error('Invalid data format: expected ArrayBuffer');
            }

            log('Processing ArrayBuffer of size:', arrayBuffer.byteLength);
            await onDataReceived(arrayBuffer);

            // Mark as received and stop listening
            hasReceivedDataRef.current = true;

            // Send acknowledgment
            sendMessage('DATA_ACK');
          } catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            log('Error processing data:', errorMessage);
            sendMessage('ERROR');
          }
          break;

        default:
          log('Unknown message type:', data.type);
      }
    }
  }, [allowedOrigins, log, onDataReceived, sendMessage]);

  /**
   * Set up message listener on mount
   */
  useEffect(() => {
    log('Setting up iframe message listener');
    window.addEventListener('message', handleMessage);

    // Mark as ready
    isReadyRef.current = true;

    // Notify parent that iframe is ready (if embedded in iframe)
    if (window.parent && window.parent !== window) {
      log('Sending READY signal to parent');
      window.parent.postMessage('READY', '*');
    }

    return () => {
      log('Cleaning up iframe message listener');
      window.removeEventListener('message', handleMessage);
      isReadyRef.current = false;
    };
  }, [handleMessage, log]);

  return {
    sendMessage,
    isReady: isReadyRef.current
  };
}
