"use client";

import dynamic from "next/dynamic";

const ArcticMap = dynamic(() => import("@/components/ArcticMap"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full flex items-center justify-center">
      <div className="text-center">
        <div className="w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
        <p className="text-gray-600">Loading Arctic Map...</p>
      </div>
    </div>
  ),
});

export default function MapPage() {
  return (
    <div className="h-full flex flex-col">
      <div className="p-6 bg-white border-b border-gray-200">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Arctic Scout Map
        </h1>
        <p className="text-gray-600">
          Real-time vessel detection and classification in Canadian Arctic
          waters
        </p>
      </div>

      <div className="flex-1 bg-gray-50">
        <ArcticMap />
      </div>
    </div>
  );
}
