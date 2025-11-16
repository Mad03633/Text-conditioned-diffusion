import React, { useState } from "react";
import Loader from "./Loader";
import GeneratedImage from "./GeneratedImage";

const DIGITS = [
  "zero", "one", "two", "three", "four",
  "five", "six", "seven", "eight", "nine"
];

export default function GeneratorForm() {
  const [text, setText] = useState("zero");
  const [loading, setLoading] = useState(false);
  const [generatedImage, setGeneratedImage] = useState(null);

  const handleGenerate = async () => {
    setLoading(true);
    setGeneratedImage(null);

    const digitIndex = DIGITS.indexOf(text);

    try {
      const response = await fetch("http://localhost:8000/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ digit: digitIndex })
      });

      const data = await response.json();

      if (data.image) {
        setGeneratedImage(`data:image/png;base64,${data.image}`);
      } else {
        alert("Error: " + data.error);
      }

    } catch (error) {
      console.error(error);
      alert("Backend is not reachable");
    }

    setLoading(false);
  };

  return (
    <div className="bg-gray-800 p-8 rounded-xl shadow-xl w-[450px]">
      <h2 className="text-2xl font-bold text-center mb-6">
        Generate a Digit
      </h2>

      <label className="block text-gray-300 mb-2">
        Choose digit description:
      </label>

      <select
        value={text}
        onChange={(e) => setText(e.target.value)}
        className="w-full p-3 rounded-lg bg-gray-700 border border-gray-600 focus:ring-2 focus:ring-indigo-500"
      >
        {DIGITS.map((d) => (
          <option key={d} value={d}>{d}</option>
        ))}
      </select>

      <button
        onClick={handleGenerate}
        className="w-full mt-6 bg-indigo-600 hover:bg-indigo-700 
                   py-3 rounded-lg font-semibold transition"
      >
        Generate
      </button>

      {loading && <Loader />}
      {generatedImage && <GeneratedImage image={generatedImage} />}
    </div>
  );
}