import { motion } from 'framer-motion';
import { FaArrowRight } from 'react-icons/fa';
import { Link } from 'react-router-dom';

function Home() {
  return (
    <div className="container mx-auto px-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-4xl mx-auto text-center"
      >
        <h1 className="text-6xl font-bold mb-6">
          Transform Your Images
        </h1>
        <p className="text-xl text-gray-400 mb-8">
          Experience the power of AI-driven super-resolution technology.
          Turn your low-resolution images into crystal-clear masterpieces.
        </p>
        <Link
          to="/demo"
          className="inline-flex items-center gap-2 bg-white text-black px-8 py-4 rounded-full hover:bg-gray-200 transition-colors"
        >
          Try Demo <FaArrowRight />
        </Link>
      </motion.div>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="mt-20 grid grid-cols-1 md:grid-cols-3 gap-8"
      >
        {[
          {
            title: "High Quality",
            description: "Get stunning high-resolution results"
          },
          {
            title: "Fast Processing",
            description: "Quick and efficient enhancement"
          },
          {
            title: "Easy to Use",
            description: "Simple drag and drop interface"
          }
        ].map((feature, index) => (
          <div
            key={index}
            className="border border-white/10 rounded-lg p-6 hover:border-white/30 transition-colors"
          >
            <h3 className="text-xl font-bold mb-2">{feature.title}</h3>
            <p className="text-gray-400">{feature.description}</p>
          </div>
        ))}
      </motion.div>
    </div>
  );
}

export default Home;