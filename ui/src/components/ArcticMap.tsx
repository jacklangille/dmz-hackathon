"use client";

import { useEffect, useRef } from "react";
import "leaflet/dist/leaflet.css";

export default function ArcticMap() {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<any>(null);

  useEffect(() => {
    if (mapRef.current && !mapInstanceRef.current) {
      // Dynamic import of Leaflet to avoid SSR issues
      import("leaflet").then((L) => {
        // Double-check that we haven't already initialized
        if (mapInstanceRef.current) {
          return;
        }

        // Fix for default markers in Next.js
        delete (L.Icon.Default.prototype as any)._getIconUrl;
        L.Icon.Default.mergeOptions({
          iconRetinaUrl:
            "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png",
          iconUrl:
            "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png",
          shadowUrl:
            "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png",
        });

        // Clear any existing map container content
        const container = mapRef.current!;
        container.innerHTML = "";

        // Remove any existing Leaflet container class that might interfere
        (container as any)._leaflet_id = null;

        // Initialize the map centered on the Arctic
        const map = L.map(container, {
          center: [69.2, -80.1], // High latitude for Arctic focus
          zoom: 3,
          minZoom: 1,
          maxZoom: 18,
        });

        // Add CartoDB Voyager tiles (more detailed with water body labels)
        L.tileLayer(
          "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
          {
            attribution:
              '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
            subdomains: "abcd",
            maxZoom: 18,
          }
        ).addTo(map);

        // Add some Arctic-specific markers for context
        const arcticLocations = [
          {
            name: "Disko Bay, Greenland",
            coords: [69.2, -51.1] as [number, number],
          },
        ];

        arcticLocations.forEach((location) => {
          L.marker(location.coords)
            .addTo(map)
            .bindPopup(`<b>${location.name}</b><br/>Arctic Region`);
        });

        // Add custom control for Arctic info

        mapInstanceRef.current = map;
      });
    }

    // Cleanup function
    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, []);

  return (
    <div
      ref={mapRef}
      className="w-full h-full"
      style={{ minHeight: "500px" }}
    />
  );
}
