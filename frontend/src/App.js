import React from "react";
import Header from "./components/Header";
import GeneratorForm from "./components/GeneratorForm";

function App() {
  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <Header />
      
      <main className="flex justify-center mt-10">
        <GeneratorForm />
      </main>
    </div>
  );
}

export default App;