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
          center: [85, -100], // High latitude for Arctic focus
          zoom: 3,
          minZoom: 2,
          maxZoom: 10,
          // Restrict the map bounds to Arctic region
          maxBounds: [
            [60, -180], // Southwest corner (southern limit)
            [90, 180], // Northeast corner (North Pole)
          ],
          maxBoundsViscosity: 1.0, // Prevents dragging outside bounds
        });

        // Add CartoDB Positron tiles (English labels, clean style)
        L.tileLayer(
          "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
          {
            attribution:
              '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
            subdomains: "abcd",
            maxZoom: 10,
          }
        ).addTo(map);

        // Add some Arctic-specific markers for context
        const arcticLocations = [
          { name: "North Pole", coords: [90, 0] as [number, number] },
          {
            name: "Svalbard, Norway",
            coords: [78.2, 15.6] as [number, number],
          },
          { name: "Alert, Canada", coords: [82.5, -62.3] as [number, number] },
          {
            name: "Barrow, Alaska",
            coords: [71.3, -156.8] as [number, number],
          },
          {
            name: "Murmansk, Russia",
            coords: [68.97, 33.08] as [number, number],
          },
          {
            name: "Reykjavik, Iceland",
            coords: [64.1, -21.9] as [number, number],
          },
        ];

        arcticLocations.forEach((location) => {
          L.marker(location.coords)
            .addTo(map)
            .bindPopup(`<b>${location.name}</b><br/>Arctic Region`);
        });

        // Add a circle to highlight the Arctic Circle
        L.circle([90, 0], {
          color: "#3b82f6",
          fillColor: "#3b82f6",
          fillOpacity: 0.1,
          radius: 2000000, // 2000km radius
          weight: 2,
        }).addTo(map);

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
