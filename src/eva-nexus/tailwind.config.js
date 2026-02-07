/** @type {import('tailwindcss').Config} */
export default {
    content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
    theme: {
        extend: {
            colors: {
                matrix: {
                    DEFAULT: '#00ff41',
                    dark: '#003b00',
                    dim: '#00cc33',
                    bright: '#33ff66',
                },
                cyber: {
                    bg: '#000000',
                    panel: '#0a0a0f',
                    surface: '#111118',
                    border: '#1a1a2e',
                    cyan: '#00d4ff',
                    pink: '#ff003c',
                    amber: '#f0a500',
                    blue: '#05d9e8',
                    purple: '#b829dd',
                }
            },
            fontFamily: {
                mono: ['"Share Tech Mono"', '"JetBrains Mono"', 'Consolas', 'monospace'],
                display: ['"Orbitron"', 'sans-serif'],
            },
            animation: {
                'glow-pulse': 'glowPulse 2s ease-in-out infinite',
                'scan': 'scan 8s linear infinite',
                'flicker': 'flicker 4s linear infinite',
                'border-flow': 'borderFlow 3s linear infinite',
                'fade-in': 'fadeIn 0.6s ease-out forwards',
                'slide-up': 'slideUp 0.5s ease-out forwards',
            },
            keyframes: {
                glowPulse: {
                    '0%, 100%': { boxShadow: '0 0 5px rgba(0,255,65,0.3)' },
                    '50%': { boxShadow: '0 0 20px rgba(0,255,65,0.8)' },
                },
                scan: {
                    '0%': { transform: 'translateY(-100%)' },
                    '100%': { transform: 'translateY(100vh)' },
                },
                flicker: {
                    '0%, 19.999%, 22%, 62.999%, 64%, 64.999%, 70%, 100%': { opacity: '1' },
                    '20%, 21.999%, 63%, 63.999%, 65%, 69.999%': { opacity: '0.4' },
                },
                borderFlow: {
                    '0%': { backgroundPosition: '0% 50%' },
                    '100%': { backgroundPosition: '200% 50%' },
                },
                fadeIn: {
                    from: { opacity: '0', transform: 'translateY(10px)' },
                    to: { opacity: '1', transform: 'translateY(0)' },
                },
                slideUp: {
                    from: { opacity: '0', transform: 'translateY(20px)' },
                    to: { opacity: '1', transform: 'translateY(0)' },
                },
            },
        }
    },
    plugins: [],
}
