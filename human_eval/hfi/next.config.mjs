/** @type {import('next').NextConfig} */

const nextConfig = {
  reactStrictMode: true,
  env: {
    PYTHON_HUMAN_EVAL_BACKEND_URL: process.env.PYTHON_HUMAN_EVAL_BACKEND_URL,
  },
  rewrites: async () => {
    return [
      {
        source: '/flask/:path*',
        destination:
          process.env.NODE_ENV === 'development'
            ? 'http://127.0.0.1:5001/:path*'
            : '/api/',
      },
    ]
  },
};

export default nextConfig;
