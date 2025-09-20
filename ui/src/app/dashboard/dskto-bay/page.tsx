"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function DiskoBayPage() {
  const [activeTab, setActiveTab] = useState("satellite");

  const tabs = ["satellite", "Red", "Green", "Blue", "Coastal", "Infrared"];

  const tabData = {
    satellite: {
      icon: "🛰️",
      title: "Satellite Imagery",
      description: "High-resolution satellite imagery",
      threatAnalysis:
        "Satellite imagery reveals unusual vessel activity in the Disko Bay region. The high-resolution data shows multiple unidentified objects that don't match typical fishing or commercial vessel patterns. Thermal signatures suggest these vessels may be operating with reduced visibility protocols.",
    },
    Red: {
      icon: "🔴",
      title: "Red Band Analysis",
      description: "Red band spectral analysis",
      threatAnalysis:
        "Red band spectral analysis indicates significant chlorophyll concentration anomalies. The data suggests potential underwater activity or environmental changes that could mask vessel detection. This spectral signature is consistent with recent intelligence reports of submarine operations in Arctic waters.",
    },
    Green: {
      icon: "🟢",
      title: "Vegetation Index",
      description: "Green band vegetation index",
      threatAnalysis:
        "Vegetation index analysis shows unexpected changes in coastal vegetation patterns. The data reveals areas where natural growth has been disturbed, potentially indicating ground-based activity or infrastructure development. These patterns are consistent with temporary encampment or equipment deployment.",
    },
    Blue: {
      icon: "🔵",
      title: "Water Analysis",
      description: "Blue band water analysis",
      threatAnalysis:
        "Blue band water analysis reveals unusual turbidity patterns and water temperature variations. The data suggests underwater activity or vessel wake patterns that don't align with normal maritime traffic. Water chemistry analysis indicates potential presence of non-native substances or equipment.",
    },
    Coastal: {
      icon: "🌊",
      title: "Coastal Monitoring",
      description: "Coastal monitoring data",
      threatAnalysis:
        "Coastal monitoring data shows irregular shoreline activity and potential landing sites. The analysis reveals disturbed sediment patterns and temporary structures that suggest recent human activity. These findings correlate with reports of unauthorized access to sensitive Arctic monitoring stations.",
    },
    Infrared: {
      icon: "🌡️",
      title: "Thermal Analysis",
      description: "Thermal infrared imagery",
      threatAnalysis:
        "Thermal infrared analysis reveals multiple heat signatures that don't correspond to known infrastructure or natural phenomena. The thermal patterns suggest active equipment or vehicles operating in the area. Temperature gradients indicate potential electronic equipment or engine activity consistent with surveillance or reconnaissance operations.",
    },
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-red-600 mb-2">
            Threat Detected - Disko Bay
          </h1>
          <p className="text-lg text-gray-700">
            A potential security threat has been identified in the Disko Bay
            region. Immediate analysis and monitoring required to assess the
            situation and determine appropriate response measures.
          </p>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Stats and Text Block */}
          <div className="space-y-6">
            {/* Location Stats */}
            <Card>
              <CardHeader>
                <CardTitle>Location Details</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="font-medium text-muted-foreground">
                      Longitude:
                    </span>
                    <span className="text-foreground">-51.1°</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="font-medium text-muted-foreground">
                      Latitude:
                    </span>
                    <span className="text-foreground">69.2°</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="font-medium text-muted-foreground">
                      Date:
                    </span>
                    <span className="text-foreground">2024-01-15</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="font-medium text-muted-foreground">
                      Time:
                    </span>
                    <span className="text-foreground">14:32:47 UTC</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Text Block */}
            <Card>
              <CardHeader>
                <CardTitle>
                  Threat Analysis -{" "}
                  {tabData[activeTab as keyof typeof tabData].title}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground leading-relaxed">
                  {tabData[activeTab as keyof typeof tabData].threatAnalysis}
                </p>
              </CardContent>
            </Card>
          </div>

          {/* Imagery Display */}
          <Card>
            <CardHeader>
              <CardTitle>
                {tabData[activeTab as keyof typeof tabData].title}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="w-full h-96 bg-muted rounded-lg flex items-center justify-center border-2 border-dashed border-muted-foreground/25">
                <div className="text-center">
                  <div className="text-6xl text-muted-foreground mb-4">
                    {tabData[activeTab as keyof typeof tabData].icon}
                  </div>
                  <p className="text-muted-foreground font-medium">
                    {tabData[activeTab as keyof typeof tabData].title}
                  </p>
                  <p className="text-sm text-muted-foreground/70 mt-2">
                    {tabData[activeTab as keyof typeof tabData].description}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Custom Tabs at Bottom */}
        <Card className="mt-8">
          <CardContent className="p-6">
            <div className="border-b border-gray-200">
              <nav className="flex space-x-8" aria-label="Tabs">
                {tabs.map((tab) => (
                  <button
                    key={tab}
                    onClick={() => setActiveTab(tab)}
                    className={`py-4 px-1 border-b-2 font-medium text-sm capitalize transition-colors ${
                      activeTab === tab
                        ? "border-red-500 text-red-600"
                        : "border-transparent text-muted-foreground hover:text-foreground hover:border-muted-foreground"
                    }`}
                  >
                    {tab}
                  </button>
                ))}
              </nav>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
