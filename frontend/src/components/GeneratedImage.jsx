import React from "react";

export default function GeneratedImage({ image }) {
  return (
    <div className="mt-6 p-4 bg-gray-800 rounded-xl shadow-xl text-center">
      <h2 className="text-xl font-semibold mb-3">Generated Image</h2>
      <img 
        src={image} 
        alt="Generated digit" 
        className="mx-auto border border-gray-600 rounded-lg"
      />
    </div>
  );
}