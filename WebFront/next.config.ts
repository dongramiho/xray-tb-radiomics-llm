import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Standalone output keeps the Docker image small (see WebFront/Dockerfile).
  output: "standalone",
};

export default nextConfig;
