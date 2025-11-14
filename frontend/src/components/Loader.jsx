import React from "react";

export default function Loader() {
  return (
    <div className="flex items-center justify-center mt-4">
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white"></div>
      <span className="ml-3 text-lg">Generating...</span>
    </div>
  );
}