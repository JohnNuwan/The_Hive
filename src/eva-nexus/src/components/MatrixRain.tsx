import { useEffect, useRef } from 'react'

export default function MatrixRain() {
    const canvasRef = useRef<HTMLCanvasElement>(null)

    useEffect(() => {
        const canvas = canvasRef.current
        if (!canvas) return

        const ctx = canvas.getContext('2d')
        if (!ctx) return

        let animationId: number

        const resize = () => {
            canvas.width = window.innerWidth
            canvas.height = window.innerHeight
        }
        resize()

        const chars = 'アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン01234567890ABCDEF<>{}[]|/\\=+*&^%$#@!'
        const fontSize = 14
        const columns = Math.floor(canvas.width / fontSize)
        const drops: number[] = new Array(columns).fill(0).map(() => Math.random() * -100)
        const speeds: number[] = new Array(columns).fill(0).map(() => 0.5 + Math.random() * 1.5)

        function draw() {
            // Fade effect
            ctx!.fillStyle = 'rgba(0, 0, 0, 0.06)'
            ctx!.fillRect(0, 0, canvas!.width, canvas!.height)

            ctx!.font = `${fontSize}px "Share Tech Mono", monospace`

            for (let i = 0; i < drops.length; i++) {
                const char = chars[Math.floor(Math.random() * chars.length)]
                const x = i * fontSize
                const y = drops[i] * fontSize

                // Head character (bright white/green)
                if (Math.random() > 0.7) {
                    ctx!.fillStyle = '#ffffff'
                    ctx!.globalAlpha = 0.9
                } else {
                    ctx!.fillStyle = '#00ff41'
                    ctx!.globalAlpha = 0.4 + Math.random() * 0.4
                }

                ctx!.fillText(char, x, y)
                ctx!.globalAlpha = 1

                // Move drop
                drops[i] += speeds[i]

                // Reset when off screen
                if (drops[i] * fontSize > canvas!.height && Math.random() > 0.98) {
                    drops[i] = 0
                }
            }

            animationId = requestAnimationFrame(draw)
        }

        draw()
        window.addEventListener('resize', resize)

        return () => {
            cancelAnimationFrame(animationId)
            window.removeEventListener('resize', resize)
        }
    }, [])

    return (
        <canvas
            ref={canvasRef}
            className="fixed inset-0 z-0 pointer-events-none"
            style={{ opacity: 0.12 }}
        />
    )
}
