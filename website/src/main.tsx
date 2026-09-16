import './monaco-setup'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { FileDiffSessionProvider } from './context/FileDiffSession.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <FileDiffSessionProvider>
      <App />
    </FileDiffSessionProvider>
  </StrictMode>,
)
