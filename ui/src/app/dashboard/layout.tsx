import Image from "next/image";
import Link from "next/link";
import scoutIcon from "@/assets/scout-icon.png";
import {
  Globe,
  AlertTriangle,
  Ship,
  BarChart3,
  Activity,
  MapPin,
  Settings,
} from "lucide-react";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen bg-gray-50 flex">
      {/* Sidebar */}
      <div className="w-20 bg-white border-r border-gray-200 flex flex-col items-center py-6 space-y-6">
        {/* Scout Icon */}
        <div className="w-12 h-12 mb-12 flex items-center justify-center">
          <Image
            src={scoutIcon}
            alt="Arctic Scout Icon"
            className="w-10 h-10"
            priority
          />
        </div>

        {/* World Icon */}
        <Link href="/dashboard/map">
          <div className="w-12 h-12 flex items-center justify-center hover:bg-blue-50 rounded-lg transition-colors duration-200 cursor-pointer">
            <Globe className="w-6 h-6 text-blue-600 hover:text-blue-700" />
          </div>
        </Link>

        {/* Exclamation Icon */}
        <div className="w-12 h-12 flex items-center justify-center hover:bg-blue-50 rounded-lg transition-colors duration-200 cursor-pointer">
          <AlertTriangle className="w-6 h-6 text-blue-600 hover:text-blue-700" />
        </div>

        {/* Vessel Intelligence Icon */}
        <div className="w-12 h-12 flex items-center justify-center hover:bg-blue-50 rounded-lg transition-colors duration-200 cursor-pointer">
          <Ship className="w-6 h-6 text-blue-600 hover:text-blue-700" />
        </div>

        {/* Analytics & Trends Icon */}
        <div className="w-12 h-12 flex items-center justify-center hover:bg-blue-50 rounded-lg transition-colors duration-200 cursor-pointer">
          <BarChart3 className="w-6 h-6 text-blue-600 hover:text-blue-700" />
        </div>

        {/* System Health & Data Status Icon */}
        <div className="w-12 h-12 flex items-center justify-center hover:bg-blue-50 rounded-lg transition-colors duration-200 cursor-pointer">
          <Activity className="w-6 h-6 text-blue-600 hover:text-blue-700" />
        </div>

        {/* Mission Planning / Export Icon */}
        <div className="w-12 h-12 flex items-center justify-center hover:bg-blue-50 rounded-lg transition-colors duration-200 cursor-pointer">
          <MapPin className="w-6 h-6 text-blue-600 hover:text-blue-700" />
        </div>

        {/* Admin & Settings Icon */}
        <div className="w-12 h-12 flex items-center justify-center hover:bg-blue-50 rounded-lg transition-colors duration-200 cursor-pointer">
          <Settings className="w-6 h-6 text-blue-600 hover:text-blue-700" />
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1">{children}</div>
    </div>
  );
}
