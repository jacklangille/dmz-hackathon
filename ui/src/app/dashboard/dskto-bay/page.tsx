import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import Image from "next/image";
import rawSatelliteImage from "@/assets/raw-satellite.png";
import waterMaskImage from "@/assets/water-mask.png";
import vesselDetectionImage from "@/assets/vessel-detection.png";
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
                  <CardContent>
                    {/* Satellite Image */}
                    <div className="flex items-center justify-center overflow-hidden border-2 border-gray-200 bg-slate-100 p-4">
                      <Image
                        src={rawSatelliteImage}
                        alt="Raw Satellite Imagery"
                        className="h-[550px] object-contain"
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
                    <div className="flex items-center justify-center overflow-hidden border-2 border-gray-200 bg-slate-100 p-4">
                      <Image
                        src={waterMaskImage}
                        alt="Water Mask"
                        className="h-[550px] object-contain"
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
                    <div className="flex items-center justify-center overflow-hidden border-2 border-gray-200 bg-slate-100 p-4">
                      <Image
                        src={vesselDetectionImage}
                        alt="Vessel Detection"
                        className="h-[550px] object-contain"
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
            <Card className="h-[800px] flex flex-col">
              <CardHeader className="pb-3 flex-shrink-0">
                <CardTitle className="text-lg flex items-center gap-2">
                  <Ship className="w-6 h-6 text-black" />
                  Detected Vessels
                </CardTitle>
              </CardHeader>
              <CardContent className="overflow-y-auto flex-1">
                <p className="text-sm mb-4 text-gray-600">
                  Vessel detection analysis from satellite imagery
                </p>
                <div className="space-y-4">
                  {[
                    {
                      id: 248,
                      position: "(5194.1, 4750.9)",
                      area: 1745,
                      aspectRatio: 3.83,
                      solidity: 0.85,
                      length: 96.5,
                      width: 25.2,
                      classification: "cargo",
                    },
                    {
                      id: 249,
                      position: "(5217.6, 4486.0)",
                      area: 37,
                      aspectRatio: 1.35,
                      solidity: 0.95,
                      length: 8.0,
                      width: 5.9,
                      classification: "fishing",
                    },
                    {
                      id: 250,
                      position: "(5235.6, 5461.7)",
                      area: 212,
                      aspectRatio: 2.58,
                      solidity: 0.84,
                      length: 27.4,
                      width: 10.6,
                      classification: "commercial",
                    },
                    {
                      id: 251,
                      position: "(5267.8, 4064.8)",
                      area: 1645,
                      aspectRatio: 2.61,
                      solidity: 0.76,
                      length: 79.5,
                      width: 30.5,
                      classification: "cargo",
                    },
                    {
                      id: 252,
                      position: "(5253.5, 5431.5)",
                      area: 28,
                      aspectRatio: 1.61,
                      solidity: 0.93,
                      length: 7.5,
                      width: 4.7,
                      classification: "fishing",
                    },
                    {
                      id: 253,
                      position: "(5271.8, 4115.1)",
                      area: 36,
                      aspectRatio: 2.04,
                      solidity: 0.86,
                      length: 10.1,
                      width: 4.9,
                      classification: "commercial",
                    },
                    {
                      id: 254,
                      position: "(5274.7, 4102.8)",
                      area: 26,
                      aspectRatio: 1.23,
                      solidity: 0.96,
                      length: 6.4,
                      width: 5.2,
                      classification: "fishing",
                    },
                    {
                      id: 255,
                      position: "(5275.7, 4014.9)",
                      area: 45,
                      aspectRatio: 2.49,
                      solidity: 0.85,
                      length: 12.3,
                      width: 4.9,
                      classification: "commercial",
                    },
                    {
                      id: 256,
                      position: "(5292.5, 5448.8)",
                      area: 131,
                      aspectRatio: 1.57,
                      solidity: 0.96,
                      length: 16.4,
                      width: 10.4,
                      classification: "fishing",
                    },
                    {
                      id: 257,
                      position: "(5298.9, 4802.3)",
                      area: 29,
                      aspectRatio: 2.33,
                      solidity: 0.88,
                      length: 9.4,
                      width: 4.0,
                      classification: "commercial",
                    },
                    {
                      id: 258,
                      position: "(5301.9, 5429.4)",
                      area: 25,
                      aspectRatio: 1.89,
                      solidity: 0.96,
                      length: 7.7,
                      width: 4.1,
                      classification: "fishing",
                    },
                    {
                      id: 259,
                      position: "(5315.0, 4877.2)",
                      area: 208,
                      aspectRatio: 2.71,
                      solidity: 0.91,
                      length: 27.3,
                      width: 10.1,
                      classification: "cargo",
                    },
                    {
                      id: 260,
                      position: "(5326.9, 5419.0)",
                      area: 162,
                      aspectRatio: 4.22,
                      solidity: 0.71,
                      length: 33.5,
                      width: 7.9,
                      classification: "commercial",
                    },
                    {
                      id: 261,
                      position: "(5332.6, 3920.9)",
                      area: 420,
                      aspectRatio: 2.62,
                      solidity: 0.67,
                      length: 42.4,
                      width: 16.2,
                      classification: "cargo",
                    },
                    {
                      id: 262,
                      position: "(5318.0, 5438.2)",
                      area: 31,
                      aspectRatio: 1.45,
                      solidity: 0.97,
                      length: 7.5,
                      width: 5.2,
                      classification: "fishing",
                    },
                    {
                      id: 263,
                      position: "(5386.9, 5402.2)",
                      area: 90,
                      aspectRatio: 2.72,
                      solidity: 0.91,
                      length: 17.9,
                      width: 6.6,
                      classification: "commercial",
                    },
                    {
                      id: 264,
                      position: "(5394.4, 3894.2)",
                      area: 100,
                      aspectRatio: 3.02,
                      solidity: 0.82,
                      length: 20.6,
                      width: 6.8,
                      classification: "cargo",
                    },
                    {
                      id: 265,
                      position: "(5419.6, 5166.0)",
                      area: 25,
                      aspectRatio: 1.53,
                      solidity: 0.96,
                      length: 6.9,
                      width: 4.5,
                      classification: "fishing",
                    },
                    {
                      id: 266,
                      position: "(5421.5, 5413.4)",
                      area: 28,
                      aspectRatio: 1.39,
                      solidity: 0.93,
                      length: 7.2,
                      width: 5.1,
                      classification: "fishing",
                    },
                    {
                      id: 267,
                      position: "(5441.4, 5443.4)",
                      area: 34,
                      aspectRatio: 1.25,
                      solidity: 0.94,
                      length: 7.4,
                      width: 6.0,
                      classification: "commercial",
                    },
                    {
                      id: 268,
                      position: "(5447.4, 4705.8)",
                      area: 52,
                      aspectRatio: 2.0,
                      solidity: 0.81,
                      length: 12.8,
                      width: 6.4,
                      classification: "commercial",
                    },
                    {
                      id: 269,
                      position: "(5449.5, 5049.9)",
                      area: 42,
                      aspectRatio: 2.16,
                      solidity: 0.95,
                      length: 10.7,
                      width: 5.0,
                      classification: "fishing",
                    },
                    {
                      id: 270,
                      position: "(5452.4, 5408.5)",
                      area: 29,
                      aspectRatio: 1.48,
                      solidity: 0.94,
                      length: 7.5,
                      width: 5.1,
                      classification: "fishing",
                    },
                    {
                      id: 271,
                      position: "(5459.9, 5276.9)",
                      area: 37,
                      aspectRatio: 2.16,
                      solidity: 0.95,
                      length: 10.1,
                      width: 4.7,
                      classification: "commercial",
                    },
                    {
                      id: 272,
                      position: "(5462.9, 5361.6)",
                      area: 37,
                      aspectRatio: 1.57,
                      solidity: 0.97,
                      length: 8.7,
                      width: 5.5,
                      classification: "fishing",
                    },
                  ].map((vessel) => (
                    <div
                      key={vessel.id}
                      className="border border-gray-200 p-3 bg-gray-50"
                    >
                      <div className="flex justify-between items-center mb-2">
                        <span className="font-semibold text-sm">
                          Ship {vessel.id}
                        </span>
                        <span className="text-xs text-gray-500">
                          Area: {vessel.area}px
                        </span>
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div>
                          <span className="text-gray-600">Position:</span>
                          <div className="font-mono">{vessel.position}</div>
                        </div>
                        <div>
                          <span className="text-gray-600">Aspect Ratio:</span>
                          <div>{vessel.aspectRatio}</div>
                        </div>
                        <div>
                          <span className="text-gray-600">Length:</span>
                          <div>{vessel.length}px</div>
                        </div>
                        <div>
                          <span className="text-gray-600">Width:</span>
                          <div>{vessel.width}px</div>
                        </div>
                        <div>
                          <span className="text-gray-600">Solidity:</span>
                          <div>{vessel.solidity}</div>
                        </div>
                        <div>
                          <span className="text-gray-600">Classification:</span>
                          <div className="mt-1">
                            <span
                              className={`px-2 py-1 text-xs ${
                                vessel.classification === "cargo"
                                  ? "bg-blue-100 text-blue-800"
                                  : vessel.classification === "commercial"
                                  ? "bg-green-100 text-green-800"
                                  : "bg-orange-100 text-orange-800"
                              }`}
                            >
                              {vessel.classification}
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
