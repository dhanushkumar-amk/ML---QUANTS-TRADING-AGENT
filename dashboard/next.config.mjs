/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  experimental: {
    useWasmBinary: true,
  },
};

export default nextConfig;
