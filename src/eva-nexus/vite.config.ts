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
            // ═══ CORE SERVICES (Triumvirat Vital) ═══
            '/api/core': {
                target: 'http://localhost:8000',
                changeOrigin: true,
                rewrite: (path) => path.replace(/^\/api\/core/, ''),
                configure: silentProxy,
            },
            '/api/banker': {
                target: 'http://localhost:8100',
                changeOrigin: true,
                rewrite: (path) => path.replace(/^\/api\/banker/, ''),
                configure: silentProxy,
            },
            '/api/kernel': {
                target: 'http://localhost:8800',
                changeOrigin: true,
                rewrite: (path) => path.replace(/^\/api\/kernel/, ''),
                configure: silentProxy,
            },
            '/api/nervous': {
                target: 'http://localhost:9090',
                changeOrigin: true,
                rewrite: (path) => path.replace(/^\/api\/nervous/, ''),
                configure: silentProxy,
            },
            // ═══ SECURITY & INTELLIGENCE ═══
            '/api/sentinel': {
                target: 'http://localhost:8200',
                changeOrigin: true,
                rewrite: (path) => path.replace(/^\/api\/sentinel/, ''),
                configure: silentProxy,
            },
            '/api/shadow': {
                target: 'http://localhost:8900',
                changeOrigin: true,
                rewrite: (path) => path.replace(/^\/api\/shadow/, ''),
                configure: silentProxy,
            },
            '/api/wraith': {
                target: 'http://localhost:9400',
                changeOrigin: true,
                rewrite: (path) => path.replace(/^\/api\/wraith/, ''),
                configure: silentProxy,
            },
            '/api/researcher': {
                target: 'http://localhost:9300',
                changeOrigin: true,
                rewrite: (path) => path.replace(/^\/api\/researcher/, ''),
                configure: silentProxy,
            },
            // ═══ SUPPORT SERVICES ═══
            '/api/compliance': {
                target: 'http://localhost:8300',
                changeOrigin: true,
                rewrite: (path) => path.replace(/^\/api\/compliance/, ''),
                configure: silentProxy,
            },
            '/api/substrate': {
                target: 'http://localhost:8400',
                changeOrigin: true,
                rewrite: (path) => path.replace(/^\/api\/substrate/, ''),
                configure: silentProxy,
            },
            '/api/accountant': {
                target: 'http://localhost:8500',
                changeOrigin: true,
                rewrite: (path) => path.replace(/^\/api\/accountant/, ''),
                configure: silentProxy,
            },
            // ═══ FACTORIES ═══
            '/api/lab': {
                target: 'http://localhost:8600',
                changeOrigin: true,
                rewrite: (path) => path.replace(/^\/api\/lab/, ''),
                configure: silentProxy,
            },
            '/api/rwa': {
                target: 'http://localhost:8700',
                changeOrigin: true,
                rewrite: (path) => path.replace(/^\/api\/rwa/, ''),
                configure: silentProxy,
            },
            '/api/builder': {
                target: 'http://localhost:9000',
                changeOrigin: true,
                rewrite: (path) => path.replace(/^\/api\/builder/, ''),
                configure: silentProxy,
            },
            '/api/muse': {
                target: 'http://localhost:9100',
                changeOrigin: true,
                rewrite: (path) => path.replace(/^\/api\/muse/, ''),
                configure: silentProxy,
            },
            '/api/sage': {
                target: 'http://localhost:9200',
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
