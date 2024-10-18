// import { StrictMode } from 'react'
// import { createRoot } from 'react-dom/client'
// import App from './App.tsx'
// import './index.css'

// createRoot(document.getElementById('root')!).render(
//   <StrictMode>
//     <App />
//   </StrictMode>,
// )



import React, { StrictMode } from 'react'
import ReactDOM from 'react-dom/client'
import { RouterProvider, createRouter } from '@tanstack/react-router'

// Import the generated route tree
import { routeTree } from './routeTree.gen'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { client as singletonClient } from './generated_types'

import './index.css'
import {NextUIProvider} from "@nextui-org/react";

// Create a new router instance
const router = createRouter({ routeTree })


singletonClient.setConfig({ baseUrl: 'http://localhost:8000' });

// Register the router instance for type safety
declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router
  }
}

const queryClient = new QueryClient()

// Render the app
const rootElement = document.getElementById('root')!
if (!rootElement.innerHTML) {
  const root = ReactDOM.createRoot(rootElement)
  root.render(
    <StrictMode>
      <QueryClientProvider client={queryClient}>
        <NextUIProvider>
        <main className="dark text-foreground bg-background min-h-screen">
          <RouterProvider router={router} />
        </main>
        </NextUIProvider>
      </QueryClientProvider>
    </StrictMode>,
  )
}
