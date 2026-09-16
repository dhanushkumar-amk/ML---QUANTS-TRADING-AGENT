"use client";

import * as React from "react";

interface Particle3D {
  x: number;
  y: number;
  z: number;
  origX: number;
  origY: number;
  origZ: number;
  isGreen: boolean;
  size: number;
  alpha: number;
}

export function ParticleHero3D() {
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null);
  const mouseRef = React.useRef({ x: 0, y: 0, targetX: 0, targetY: 0 });

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationFrameId: number;
    let width = (canvas.width = canvas.parentElement?.clientWidth || 800);
    let height = (canvas.height = canvas.parentElement?.clientHeight || 400);

    const handleResize = () => {
      if (!canvas || !canvas.parentElement) return;
      width = canvas.width = canvas.parentElement.clientWidth;
      height = canvas.height = canvas.parentElement.clientHeight;
    };
    window.addEventListener("resize", handleResize);

    const handleMouseMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const x = (e.clientX - rect.left) / width - 0.5;
      const y = (e.clientY - rect.top) / height - 0.5;
      mouseRef.current.targetX = x * 1.5;
      mouseRef.current.targetY = y * 1.2;
    };
    window.addEventListener("mousemove", handleMouseMove);

    // Generate 3D Particle Cloud forming Candlestick & Market Waves
    const particles: Particle3D[] = [];
    const NUM_CANDLES = 12;
    const CANDLE_SPACING = 55;

    for (let c = 0; c < NUM_CANDLES; c++) {
      const cx = (c - NUM_CANDLES / 2) * CANDLE_SPACING;
      const cyBase = Math.sin(c * 0.6) * 40;
      const candleHeight = 30 + Math.abs(Math.cos(c * 0.8)) * 50;
      const isGreen = c % 3 !== 0;

      // Candle body particles
      for (let i = 0; i < 22; i++) {
        const px = cx + (Math.random() - 0.5) * 22;
        const py = cyBase + (Math.random() - 0.5) * candleHeight;
        const pz = (Math.random() - 0.5) * 120;
        particles.push({
          x: px,
          y: py,
          z: pz,
          origX: px,
          origY: py,
          origZ: pz,
          isGreen,
          size: 1.5 + Math.random() * 1.5,
          alpha: 0.3 + Math.random() * 0.7,
        });
      }

      // Wick particles
      for (let i = 0; i < 10; i++) {
        const py = cyBase + (Math.random() - 0.5) * (candleHeight + 45);
        const pz = (Math.random() - 0.5) * 20;
        particles.push({
          x: cx,
          y: py,
          z: pz,
          origX: cx,
          origY: py,
          origZ: pz,
          isGreen,
          size: 1.2,
          alpha: 0.4 + Math.random() * 0.5,
        });
      }
    }

    // Ambient floating grid particles
    for (let i = 0; i < 120; i++) {
      const px = (Math.random() - 0.5) * 800;
      const py = 60 + Math.random() * 80;
      const pz = (Math.random() - 0.5) * 600;
      particles.push({
        x: px,
        y: py,
        z: pz,
        origX: px,
        origY: py,
        origZ: pz,
        isGreen: false,
        size: 1,
        alpha: 0.15 + Math.random() * 0.3,
      });
    }

    let angleY = 0;

    const render = () => {
      // Smooth mouse interpolation
      mouseRef.current.x += (mouseRef.current.targetX - mouseRef.current.x) * 0.05;
      mouseRef.current.y += (mouseRef.current.targetY - mouseRef.current.y) * 0.05;

      angleY += 0.003;
      const rotY = angleY + mouseRef.current.x * 0.8;
      const rotX = mouseRef.current.y * 0.6;

      ctx.clearRect(0, 0, width, height);

      const fov = 380;
      const centerX = width / 2;
      const centerY = height / 2 + 10;

      // Project and draw particles
      const projected = particles.map((p) => {
        // Rotate Y
        const cosY = Math.cos(rotY);
        const sinY = Math.sin(rotY);
        const x1 = p.origX * cosY - p.origZ * sinY;
        const z1 = p.origZ * cosY + p.origX * sinY;

        // Rotate X
        const cosX = Math.cos(rotX);
        const sinX = Math.sin(rotX);
        const y2 = p.origY * cosX - z1 * sinX;
        const z2 = z1 * cosX + p.origY * sinX;

        const depth = z2 + 450;
        if (depth <= 0) return null;

        const scale = fov / depth;
        const screenX = centerX + x1 * scale;
        const screenY = centerY + y2 * scale;

        return {
          x: screenX,
          y: screenY,
          scale,
          depth,
          isGreen: p.isGreen,
          size: p.size * scale,
          alpha: p.alpha * Math.min(1, Math.max(0.1, scale * 1.1)),
        };
      });

      // Sort by depth (painter's algorithm)
      projected.sort((a, b) => (b?.depth || 0) - (a?.depth || 0));

      // Draw particle connections (wireframe data strands)
      ctx.lineWidth = 0.5;
      for (let i = 0; i < projected.length; i += 4) {
        const p1 = projected[i];
        const p2 = projected[i + 1];
        if (p1 && p2) {
          const dist = Math.hypot(p1.x - p2.x, p1.y - p2.y);
          if (dist < 55) {
            ctx.strokeStyle = p1.isGreen
              ? `rgba(0, 179, 134, ${p1.alpha * 0.25})`
              : `rgba(255, 255, 255, ${p1.alpha * 0.12})`;
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.stroke();
          }
        }
      }

      // Draw particle nodes
      for (const p of projected) {
        if (!p) continue;
        ctx.beginPath();
        ctx.arc(p.x, p.y, Math.max(0.8, p.size), 0, Math.PI * 2);
        if (p.isGreen) {
          ctx.fillStyle = `rgba(0, 179, 134, ${p.alpha})`;
        } else {
          ctx.fillStyle = `rgba(240, 240, 240, ${p.alpha * 0.7})`;
        }
        ctx.fill();
      }

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener("resize", handleResize);
      window.removeEventListener("mousemove", handleMouseMove);
    };
  }, []);

  return (
    <div className="relative w-full h-[320px] sm:h-[380px] overflow-hidden flex items-center justify-center">
      <canvas
        ref={canvasRef}
        className="w-full h-full cursor-crosshair opacity-90 transition-opacity hover:opacity-100"
      />
      {/* Subtle radial vignette gradient to blend canvas edges */}
      <div className="absolute inset-0 pointer-events-none bg-[radial-gradient(ellipse_at_center,transparent_40%,#080808_95%)]" />
    </div>
  );
}
