"use client";

import { useEffect, useRef } from "react";
import "leaflet/dist/leaflet.css";

// CSS for pulsing animation
const pulseStyles = `
  .pulse-marker-container {
    background: transparent !important;
    border: none !important;
  }
  
  .pulse-marker {
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background-color: #ef4444;
    position: relative;
    animation: pulse 2s infinite;
    box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7);
  }
  
  .pulse-marker::before,
  .pulse-marker::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    border-radius: 50%;
    background-color: rgba(239, 68, 68, 0.6);
    animation: pulse-ring 2s infinite;
  }
  
  .pulse-marker::after {
    animation-delay: 1s;
  }
  
  @keyframes pulse {
    0% {
      transform: scale(1);
      opacity: 1;
    }
    50% {
      transform: scale(1.1);
      opacity: 0.9;
    }
    100% {
      transform: scale(1);
      opacity: 1;
    }
  }
  
  @keyframes pulse-ring {
    0% {
      transform: scale(1);
      opacity: 0.6;
    }
    100% {
      transform: scale(2.5);
      opacity: 0;
    }
  }
`;

export default function ArcticMap() {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<any>(null);

  useEffect(() => {
    if (mapRef.current && !mapInstanceRef.current) {
      // Inject CSS styles
      const styleElement = document.createElement("style");
      styleElement.setAttribute("data-pulse-marker", "true");
      styleElement.textContent = pulseStyles;
      document.head.appendChild(styleElement);

      // Dynamic import of Leaflet to avoid SSR issues
      import("leaflet").then((L) => {
        // Double-check that we haven't already initialized
        if (mapInstanceRef.current) {
          return;
        }

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
          zoomControl: false, // Disable default zoom control to reposition it
        });

        // Add zoom control to top right
        L.control
          .zoom({
            position: "topright",
          })
          .addTo(map);

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

        // Create custom pulsing icon
        const pulseIcon = L.divIcon({
          className: "pulse-marker-container",
          html: '<div class="pulse-marker"></div>',
          iconSize: [20, 20],
          iconAnchor: [10, 10],
          popupAnchor: [0, -10],
        });

        arcticLocations.forEach((location) => {
          L.marker(location.coords, { icon: pulseIcon })
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
      // Clean up injected styles
      const existingStyle = document.querySelector("style[data-pulse-marker]");
      if (existingStyle) {
        existingStyle.remove();
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
