import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion } from 'framer-motion';
import { FaUpload, FaSpinner } from 'react-icons/fa';

function Demo() {
  const [inputImage, setInputImage] = useState(null);
  const [outputImage, setOutputImage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      setInputImage(URL.createObjectURL(file));
      setIsLoading(true);

      const formData = new FormData();
      formData.append('file', file);

      try {
        const response = await fetch('https://0d36-34-143-253-96.ngrok-free.app/enhance', {
          method: 'POST',
          body: formData,
        });

        if (response.ok) {
          const blob = await response.blob();
          setOutputImage(URL.createObjectURL(blob));
        } else {
          console.error('Error enhancing image');
        }
      } catch (error) {
        console.error('Error:', error);
      } finally {
        setIsLoading(false);
      }
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png']
    },
    multiple: false
  });

  return (
    <div className="container mx-auto px-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-6xl mx-auto"
      >
        <h2 className="text-4xl font-bold mb-8 text-center">Image Enhancement Demo</h2>

        {!inputImage && (
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-colors
              ${isDragActive ? 'border-white' : 'border-white/30 hover:border-white/60'}`}
          >
            <input {...getInputProps()} />
            <FaUpload className="mx-auto text-4xl mb-4" />
            <p className="text-xl">
              {isDragActive
                ? "Drop your image here"
                : "Drag & drop an image here, or click to select"}
            </p>
            <p className="text-sm text-gray-400 mt-2">
              Supports JPG, JPEG, PNG
            </p>
          </div>
        )}

        {inputImage && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="border border-white/10 rounded-lg p-4">
              <h3 className="text-xl font-bold mb-4">Input Image</h3>
              <img
                src={inputImage}
                alt="Input"
                className="w-full h-auto rounded-lg"
              />
            </div>

            <div className="border border-white/10 rounded-lg p-4">
              <h3 className="text-xl font-bold mb-4">Enhanced Result</h3>
              {isLoading ? (
                <div className="flex items-center justify-center h-64">
                  <FaSpinner className="animate-spin text-4xl" />
                </div>
              ) : outputImage ? (
                <img
                  src={outputImage}
                  alt="Output"
                  className="w-full h-auto rounded-lg"
                />
              ) : (
                <div className="flex items-center justify-center h-64 text-gray-400">
                  Processing...
                </div>
              )}
            </div>
          </div>
        )}

        {inputImage && (
          <button
            onClick={() => {
              setInputImage(null);
              setOutputImage(null);
            }}
            className="mt-8 px-6 py-3 bg-white text-black rounded-full hover:bg-gray-200 transition-colors"
          >
            Try Another Image
          </button>
        )}
      </motion.div>
    </div>
  );
}

export default Demo;
