import { Button } from "@/components/ui/button";
import Image from "next/image";
import scoutIcon from "@/assets/scout-icon.png";
import { ArrowUpRight } from "lucide-react";
import Link from "next/link";

export default function Home() {
  return (
    <div className="min-h-screen bg-white flex flex-col items-center justify-center p-8">
      <div className="max-w-4xl w-full space-y-8">
        {/* Header */}
        <div className="text-center">
          <Image
            src={scoutIcon}
            alt="Arctic Scout Icon"
            className="w-24 h-24 mx-auto mb-8"
            priority
          />
          <h1 className="text-6xl font-bold text-gray-900 mb-6">
            Arctic Scout
          </h1>
          <p className="text-2xl text-gray-600 mb-12">
            Advanced vessel detection and classification for Canadian Arctic
            waters
          </p>

          {/* View Application Button */}
          <Link href="/dashboard/map">
            <Button
              variant="default"
              size="lg"
              className="px-8 py-2 text-xl font-semibold"
            >
              View Application
              <div className="flex items-center justify-center ml-4">
                <ArrowUpRight size={12} />
              </div>
            </Button>
          </Link>
        </div>
      </div>
    </div>
  );
}
