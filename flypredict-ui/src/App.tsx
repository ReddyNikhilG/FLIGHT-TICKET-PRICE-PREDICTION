import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import LandingPage from "./components/LandingPage";
import LoginPage from "./components/LoginPage";
import PredictPage from "./components/PredictPage";
import { AuthProvider, useAuth } from "./context/AuthContext";

function AppRoutes() {
  const { isLoggedIn } = useAuth();

  return (
    <Routes>
      <Route path="/" element={isLoggedIn ? <LandingPage /> : <LoginPage />} />
      <Route path="/predict" element={isLoggedIn ? <PredictPage /> : <Navigate to="/" replace />} />
      <Route path="/login" element={isLoggedIn ? <Navigate to="/" replace /> : <LoginPage />} />
      <Route path="/landing" element={isLoggedIn ? <LandingPage /> : <Navigate to="/" replace />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <AppRoutes />
      </BrowserRouter>
    </AuthProvider>
  );
}

export default App;