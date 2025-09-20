import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import Image from "next/image";
import rawSatelliteImage from "@/assets/raw-satellite.png";
import waterMaskImage from "@/assets/water-mask.png";
import vesselDetectionImage from "@/assets/icebergs.png";
import {
  Satellite,
  Circle,
  Leaf,
  Waves,
  MapPin,
  Thermometer,
  Ship,
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
  ];

  return (
    <div className="bg-gray-50 h-screen flex flex-col">
      {/* Header */}
      <div className="p-6 bg-white border-b border-gray-200">
        <h1 className="text-2xl font-bold text-gray-900 mb-2">
          Threat Detected - Disko Bay
        </h1>
        <p className="text-gray-600 flex items-center gap-2">
          <span className="w-3 h-3 bg-red-500 rounded-full"></span>
          Threat Level: HIGH
        </p>
      </div>

      {/* Main Content */}
      <div className="p-6 flex-1 overflow-y-auto h-full">
        {/* Analysis Data */}
        <div className="grid grid-cols-3 gap-6 h-full">
          <div className="col-span-2 flex flex-col gap-4 h-full">
            <h2 className="text-2xl font-bold text-gray-900">Analysis Data</h2>

            <Tabs defaultValue="satellite" className="w-full flex-1">
              <TabsContent value="satellite">
                <Card className="h-full flex flex-col">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg flex items-center gap-2">
                      <Satellite className="w-6 h-6 text-black" />
                      Satellite Imagery
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="p-0 h-full flex-1 flex items-center justify-center overflow-hidden">
                    {/* Satellite Image */}
                    <div className="flex items-center justify-center overflow-hidden">
                      <Image
                        src={rawSatelliteImage}
                        alt="Raw Satellite Imagery"
                        className="h-[600px] object-contain"
                      />
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="water-mask">
                <Card className="h-full">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg flex items-center gap-2">
                      <Circle className="w-6 h-6 text-black" />
                      Water Mask
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-center justify-center overflow-hidden">
                      <Image
                        src={waterMaskImage}
                        alt="Water Mask"
                        className="h-[600px] object-contain"
                      />
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="vessel-detection">
                <Card className="h-full">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg flex items-center gap-2">
                      <Leaf className="w-6 h-6 text-black" />
                      Vessel Detection
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-center justify-center overflow-hidden border-2 border-gray-200 rounded-lg bg-gray-50">
                      <Image
                        src={vesselDetectionImage}
                        alt="Vessel Detection"
                        className="h-[600px] object-contain"
                      />
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
              <TabsList className="grid w-full grid-cols-3 mt-4">
                <TabsTrigger value="satellite">Satellite Imagery</TabsTrigger>
                <TabsTrigger value="water-mask">Water Mask</TabsTrigger>
                <TabsTrigger value="vessel-detection">
                  Vessel Detection
                </TabsTrigger>
              </TabsList>
            </Tabs>
          </div>

          {/* Vessel Information Card */}
          <div className="flex flex-col gap-4">
            <h2 className="text-2xl font-bold text-gray-900">
              Vessel Intelligence
            </h2>
            <Card className="h-full flex flex-col">
              <CardHeader className="pb-3">
                <CardTitle className="text-lg flex items-center gap-2">
                  <Ship className="w-6 h-6 text-black" />
                  Vessel Information
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm mb-8">
                  AIS vessel identification & location data
                </p>
                <div className="space-y-6">
                  <div className="flex justify-between">
                    <span className="text-sm font-bold">Longitude:</span>
                    <span className="text-sm">-51.1°</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm font-bold">Latitude:</span>
                    <span className="text-sm ">69.2°</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm font-bold">Date:</span>
                    <span className="text-sm">2024-01-15</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm font-bold">Time:</span>
                    <span className="text-sm">14:32:47 UTC</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm font-bold">
                      Vessel Classification:
                    </span>
                    <span className="text-sm">Unknown</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm font-bold">Dark Vessel:</span>
                    <span className="text-sm">Yes</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm font-bold">Distance:</span>
                    <span className="text-sm">2.3 km</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
