import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Magic Table",
  description: "Voice-controlled object retrieval — web interface",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-white text-black min-h-screen">{children}</body>
    </html>
  );
}
