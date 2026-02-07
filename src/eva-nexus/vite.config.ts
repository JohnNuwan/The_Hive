import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Suppress ECONNREFUSED proxy errors when backend services are not running
const silentProxy = (proxy: any) => {
    proxy.on('error', (_err: any, _req: any, res: any) => {
        if (res && res.writeHead) {
            res.writeHead(502, { 'Content-Type': 'application/json' })
            res.end(JSON.stringify({ error: 'Service unavailable' }))
        }
    })
}

export default defineConfig({
    plugins: [react()],
    server: {
        port: 3001,
        proxy: {
            // ═══ CORE SERVICES ═══
            '/api/core': {
                target: 'http://localhost:8000',
                changeOrigin: true,
                rewrite: (path) => path.replace(/^\/api\/core/, ''),
                configure: silentProxy,
            },
            '/api/banker': {
                target: 'http://localhost:8001',
                changeOrigin: true,
                rewrite: (path) => path.replace(/^\/api\/banker/, ''),
                configure: silentProxy,
            },
            '/api/kernel': {
                target: 'http://localhost:8080',
                changeOrigin: true,
                rewrite: (path) => path.replace(/^\/api\/kernel/, ''),
                configure: silentProxy,
            },
            '/api/nervous': {
                target: 'http://localhost:8081',
                changeOrigin: true,
                rewrite: (path) => path.replace(/^\/api\/nervous/, ''),
                configure: silentProxy,
            },
            // ═══ OSINT & SECURITY ═══
            '/api/shadow': {
                target: 'http://localhost:8002',
                changeOrigin: true,
                rewrite: (path) => path.replace(/^\/api\/shadow/, ''),
                configure: silentProxy,
            },
            '/api/sentinel': {
                target: 'http://localhost:8007',
                changeOrigin: true,
                rewrite: (path) => path.replace(/^\/api\/sentinel/, ''),
                configure: silentProxy,
            },
            '/api/wraith': {
                target: 'http://localhost:8012',
                changeOrigin: true,
                rewrite: (path) => path.replace(/^\/api\/wraith/, ''),
                configure: silentProxy,
            },
            '/api/researcher': {
                target: 'http://localhost:8013',
                changeOrigin: true,
                rewrite: (path) => path.replace(/^\/api\/researcher/, ''),
                configure: silentProxy,
            },
            // ═══ FACTORIES & SUPPORT ═══
            '/api/builder': {
                target: 'http://localhost:8003',
                changeOrigin: true,
                rewrite: (path) => path.replace(/^\/api\/builder/, ''),
                configure: silentProxy,
            },
            '/api/lab': {
                target: 'http://localhost:8004',
                changeOrigin: true,
                rewrite: (path) => path.replace(/^\/api\/lab/, ''),
                configure: silentProxy,
            },
            '/api/muse': {
                target: 'http://localhost:8005',
                changeOrigin: true,
                rewrite: (path) => path.replace(/^\/api\/muse/, ''),
                configure: silentProxy,
            },
            '/api/rwa': {
                target: 'http://localhost:8006',
                changeOrigin: true,
                rewrite: (path) => path.replace(/^\/api\/rwa/, ''),
                configure: silentProxy,
            },
            '/api/compliance': {
                target: 'http://localhost:8008',
                changeOrigin: true,
                rewrite: (path) => path.replace(/^\/api\/compliance/, ''),
                configure: silentProxy,
            },
            '/api/accountant': {
                target: 'http://localhost:8009',
                changeOrigin: true,
                rewrite: (path) => path.replace(/^\/api\/accountant/, ''),
                configure: silentProxy,
            },
            '/api/substrate': {
                target: 'http://localhost:8010',
                changeOrigin: true,
                rewrite: (path) => path.replace(/^\/api\/substrate/, ''),
                configure: silentProxy,
            },
            '/api/sage': {
                target: 'http://localhost:8011',
                changeOrigin: true,
                rewrite: (path) => path.replace(/^\/api\/sage/, ''),
                configure: silentProxy,
            },
        },
    },
    build: {
        outDir: 'dist',
        sourcemap: true,
    },
})
