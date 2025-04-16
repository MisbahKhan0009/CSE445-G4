import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { FaHome, FaImage } from 'react-icons/fa';
import Home from './pages/Home';
import Demo from './pages/Demo';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-black text-white">
        <nav className="fixed top-0 w-full bg-black/50 backdrop-blur-lg border-b border-white/10 z-50">
          <div className="container mx-auto px-4 py-4">
            <div className="flex justify-between items-center">
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                className="text-2xl font-bold"
              >
                SuperRes
              </motion.div>
              <div className="flex gap-6">
                <Link to="/" className="flex items-center gap-2 hover:text-gray-300 transition-colors">
                  <FaHome /> Home
                </Link>
                <Link to="/demo" className="flex items-center gap-2 hover:text-gray-300 transition-colors">
                  <FaImage /> Demo
                </Link>
              </div>
            </div>
          </div>
        </nav>
        <div className="pt-20">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/demo" element={<Demo />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;