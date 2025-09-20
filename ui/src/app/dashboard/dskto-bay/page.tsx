import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Satellite,
  Circle,
  Leaf,
  Waves,
  MapPin,
  Thermometer,
} from "lucide-react";

export default function DiskoBayPage() {
  const satelliteBands = [
    {
      icon: Satellite,
      title: "Satellite Imagery",
      description: "High-resolution satellite imagery",
      analysis:
        "Satellite imagery reveals unusual vessel activity in the Disko Bay region. The high-resolution data shows multiple unidentified objects that don't match typical fishing or commercial vessel patterns.",
    },
    {
      icon: Circle,
      title: "Red Band Analysis",
      description: "Red band spectral analysis",
      analysis:
        "Red band spectral analysis indicates significant chlorophyll concentration anomalies. The data suggests potential underwater activity or environmental changes that could mask vessel detection.",
    },
    {
      icon: Leaf,
      title: "Vegetation Index",
      description: "Green band vegetation index",
      analysis:
        "Vegetation index analysis shows unexpected changes in coastal vegetation patterns. The data reveals areas where natural growth has been disturbed, potentially indicating ground-based activity.",
    },
    {
      icon: Waves,
      title: "Water Analysis",
      description: "Blue band water analysis",
      analysis:
        "Blue band water analysis reveals unusual turbidity patterns and water temperature variations. The data suggests underwater activity or vessel wake patterns that don't align with normal maritime traffic.",
    },
    {
      icon: MapPin,
      title: "Coastal Monitoring",
      description: "Coastal monitoring data",
      analysis:
        "Coastal monitoring data shows irregular shoreline activity and potential landing sites. The analysis reveals disturbed sediment patterns and temporary structures that suggest recent human activity.",
    },
    {
      icon: Thermometer,
      title: "Thermal Analysis",
      description: "Thermal infrared imagery",
      analysis:
        "Thermal infrared analysis reveals multiple heat signatures that don't correspond to known infrastructure or natural phenomena. The thermal patterns suggest active equipment or vehicles operating in the area.",
    },
  ];

  return (
    <div className="min-h-screen bg-gray-50 p-6">
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
      {/* Location Details and Threat Level */}
      <div className="space-y-6 grid grid-cols-2 gap-8 mb-8">
        {/* Location Stats */}
        <Card className="h-full">
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
                <span className="font-medium text-muted-foreground">Date:</span>
                <span className="text-foreground">2024-01-15</span>
              </div>
              <div className="flex justify-between">
                <span className="font-medium text-muted-foreground">Time:</span>
                <span className="text-foreground">14:32:47 UTC</span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Threat Level Card */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <span className="w-3 h-3 bg-red-500 rounded-full"></span>
              Threat Level: HIGH
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground leading-relaxed">
              <strong>HIGH Threat Level</strong> indicates immediate security
              concerns requiring urgent attention. This classification is
              assigned when multiple intelligence sources confirm suspicious
              activity, unusual vessel patterns, or potential unauthorized
              operations in sensitive Arctic regions. Immediate response
              protocols are activated, and continuous monitoring is required
              until the threat is neutralized or confirmed as non-hostile.
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Satellite Imagery Grid */}
      <div className="space-y-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          Satellite Imagery Analysis
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {satelliteBands.map((band, index) => (
            <Card key={index} className="hover:shadow-lg transition-shadow">
              <CardHeader className="pb-3">
                <CardTitle className="text-lg flex items-center gap-2">
                  <band.icon className="w-6 h-6 text-black" />
                  {band.title}
                </CardTitle>
              </CardHeader>
              <CardContent>
                {/* Image Placeholder */}
                <div className="w-full h-32 bg-muted rounded-lg flex items-center justify-center border-2 border-dashed border-muted-foreground/25 mb-4">
                  <div className="text-center">
                    <band.icon className="w-8 h-8 text-muted-foreground mb-2" />
                    <p className="text-xs text-muted-foreground">
                      {band.title} Image
                    </p>
                  </div>
                </div>

                <p className="text-sm text-muted-foreground mb-3">
                  {band.description}
                </p>
                <p className="text-xs text-muted-foreground leading-relaxed">
                  {band.analysis}
                </p>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}
