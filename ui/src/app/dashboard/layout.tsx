import Image from "next/image";
import scoutIcon from "@/assets/scout-icon.png";
import { Globe, AlertTriangle } from "lucide-react";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen bg-gray-50 flex">
      {/* Sidebar */}
      <div className="w-20 bg-white border-r border-gray-200 flex flex-col items-center py-6 space-y-8">
        {/* Scout Icon */}
        <div className="w-12 h-12 flex items-center justify-center">
          <Image
            src={scoutIcon}
            alt="Arctic Scout Icon"
            className="w-10 h-10"
            priority
          />
        </div>

        {/* World Icon */}
        <div className="w-12 h-12 flex items-center justify-center hover:bg-blue-50 rounded-lg transition-colors duration-200 cursor-pointer">
          <Globe className="w-6 h-6 text-blue-600 hover:text-blue-700" />
        </div>

        {/* Exclamation Icon */}
        <div className="w-12 h-12 flex items-center justify-center hover:bg-blue-50 rounded-lg transition-colors duration-200 cursor-pointer">
          <AlertTriangle className="w-6 h-6 text-blue-600 hover:text-blue-700" />
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1">{children}</div>
    </div>
  );
}
