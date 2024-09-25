import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Script from "next/script";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Human Feedback Interface",
  description: "Finetuning models with human feedback",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <>
      <html lang="en">
      <head>
          <title>Open-Instruct Human Evaluation</title>
        </head>
        <body className={inter.className}>{children}</body>
        <Script src="./node_modules/preline/dist/preline.js"></Script>
      </html>
    </>
  );
}
