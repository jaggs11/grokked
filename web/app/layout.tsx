import type { Metadata } from "next";
import { Spectral, IBM_Plex_Mono, Archivo } from "next/font/google";
import "./globals.css";

// BLUEPRINT §1.3 type stack
const spectral = Spectral({ subsets: ["latin"], weight: ["400","600"], variable: "--font-spectral" });
const mono = IBM_Plex_Mono({ subsets: ["latin"], weight: ["400","500"], variable: "--font-mono" });
const archivo = Archivo({ subsets: ["latin"], weight: ["500","600"], variable: "--font-archivo" });

export const metadata: Metadata = {
  title: "Grokked — when a network stops memorizing and starts understanding",
  description: "An interactive reconstruction of grokking, from real model checkpoints.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${spectral.variable} ${mono.variable} ${archivo.variable}`}>
      <body>{children}</body>
    </html>
  );
}
