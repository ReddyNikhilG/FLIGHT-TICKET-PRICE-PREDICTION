import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { useAuth } from "../context/AuthContext";
import flightImg from "../assets/flight.png";

const LoginPage = () => {
  const navigate = useNavigate();
  const { login } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  const handleLogin = () => {
    if (!email.trim() || !password.trim()) {
      setError("Enter both email and password to continue.");
      return;
    }

    const user = email.trim();
    login(user);
    navigate("/landing", { replace: true });
  };

  const cardVariants = {
    hidden: { opacity: 0, y: 40 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.5,
        ease: "easeOut",
        when: "beforeChildren",
        staggerChildren: 0.08,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 18 },
    visible: { opacity: 1, y: 0 },
  };

  const inputContainerVariants = {
    hidden: {},
    visible: {
      transition: {
        staggerChildren: 0.08,
      },
    },
  };

  const leftPanelVariants = {
    hidden: { opacity: 0, x: -50 },
    visible: { opacity: 1, x: 0, transition: { duration: 0.8, ease: "easeOut" } },
  };

  const inputClassName =
    "mt-2 w-full rounded-xl border border-gray-300/90 bg-white/95 px-4 py-3 text-sm text-slate-800 shadow-sm outline-none transition-all duration-200 focus:border-indigo-300 focus:ring-2 focus:ring-indigo-500/70 dark:border-white/15 dark:bg-black/35 dark:text-slate-100 dark:placeholder:text-slate-400 dark:focus:border-indigo-400/60";

  return (
    <div className="min-h-screen grid md:grid-cols-2 bg-gradient-to-br from-indigo-50 via-white to-blue-50 dark:from-gray-900 dark:to-black">
      <motion.section
        variants={leftPanelVariants}
        initial="hidden"
        animate="visible"
        className="relative hidden overflow-hidden md:block"
      >
        <div className="absolute inset-0 bg-cover bg-center" style={{ backgroundImage: `url(${flightImg})` }} />
        <div className="absolute inset-0 bg-gradient-to-b from-black/70 via-black/50 to-black/80" />
        <motion.div
          aria-hidden="true"
          animate={{ x: ["-8%", "8%", "-8%"], y: ["0%", "4%", "0%"] }}
          transition={{ duration: 14, repeat: Infinity, ease: "easeInOut" }}
          className="absolute inset-0 bg-[radial-gradient(circle_at_20%_20%,rgba(99,102,241,0.35),transparent_45%),radial-gradient(circle_at_80%_70%,rgba(56,189,248,0.24),transparent_50%)]"
        />
        <motion.div
          aria-hidden="true"
          animate={{ opacity: [0.25, 0.45, 0.25] }}
          transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
          className="absolute inset-0 bg-gradient-to-tr from-indigo-500/15 via-transparent to-sky-400/10"
        />
        <motion.div
          animate={{ y: [0, -10, 0] }}
          transition={{ repeat: Infinity, duration: 4, ease: "easeInOut" }}
          className="absolute -left-14 top-20 h-44 w-44 rounded-full bg-indigo-500/25 blur-xl"
        />
        <motion.div
          animate={{ y: [0, 8, 0] }}
          transition={{ repeat: Infinity, duration: 5, ease: "easeInOut" }}
          className="absolute bottom-20 right-10 h-56 w-56 rounded-full bg-sky-400/25 blur-xl"
        />
        <motion.div
          aria-hidden="true"
          animate={{ opacity: [0.3, 0.6, 0.3], scale: [1, 1.06, 1] }}
          transition={{ duration: 7, repeat: Infinity, ease: "easeInOut" }}
          className="absolute left-1/2 top-1/2 h-[26rem] w-[26rem] -translate-x-1/2 -translate-y-1/2 rounded-full bg-indigo-500/20 blur-3xl"
        />
        <div className="pointer-events-none absolute inset-0 opacity-35" style={{ backgroundImage: "radial-gradient(rgba(255,255,255,0.28) 1px, transparent 1px)", backgroundSize: "20px 20px" }} />

        <motion.div
          animate={{ y: [0, -12, 0] }}
          transition={{ repeat: Infinity, duration: 4, ease: "easeInOut" }}
          className="relative z-10 flex h-full flex-col justify-end px-12 pb-14 space-y-4"
        >
          <p className="inline-flex w-fit rounded-full border border-white/20 bg-white/10 px-4 py-1 text-xs font-semibold uppercase tracking-[0.2em] text-slate-100 backdrop-blur-sm">
            AI Flight Intelligence
          </p>
          <p className="text-5xl font-extrabold tracking-tight text-white [text-shadow:0_8px_28px_rgba(0,0,0,0.45)]">FlyPredict</p>
          <p className="text-xl text-gray-200 [text-shadow:0_6px_20px_rgba(0,0,0,0.35)]">Predict flight prices before you book</p>
          <p className="max-w-md text-slate-200/95 leading-relaxed [text-shadow:0_4px_16px_rgba(0,0,0,0.3)]">
            AI-powered insights to help you save up to 40% on flights.
          </p>
        </motion.div>
      </motion.section>

      <section className="relative flex items-center justify-center px-6 py-8 md:px-10">
        <motion.div
          aria-hidden="true"
          animate={{ y: [0, -14, 0], x: [0, 8, 0] }}
          transition={{ duration: 9, repeat: Infinity, ease: "easeInOut" }}
          className="pointer-events-none absolute -top-24 -left-16 h-72 w-72 rounded-full bg-indigo-300/40 blur-3xl dark:bg-indigo-500/25"
        />
        <motion.div
          aria-hidden="true"
          animate={{ y: [0, 16, 0], x: [0, -10, 0] }}
          transition={{ duration: 11, repeat: Infinity, ease: "easeInOut" }}
          className="pointer-events-none absolute -bottom-28 -right-20 h-80 w-80 rounded-full bg-blue-300/35 blur-3xl dark:bg-sky-500/20"
        />

        <motion.div
          variants={cardVariants}
          initial="hidden"
          animate="visible"
          className="relative w-full max-w-md rounded-2xl border border-gray-200 bg-white/80 p-8 shadow-2xl backdrop-blur-xl dark:border-white/10 dark:bg-white/5"
        >
          <motion.div variants={itemVariants} className="mb-8 text-center">
            <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-indigo-600 text-lg font-bold text-white shadow-lg shadow-indigo-600/40">
              ✈
            </div>
            <p className="text-xs font-semibold uppercase tracking-[0.22em] text-indigo-600 dark:text-indigo-300">FlyPredict</p>
            <h1 className="mt-3 text-3xl font-bold tracking-tight text-gray-900 dark:text-white">Welcome Back</h1>
            <p className="mt-2 text-sm leading-relaxed text-gray-600 dark:text-slate-300">
              Login to continue predicting flight prices
            </p>
          </motion.div>

          <motion.div variants={itemVariants} className="mb-6 rounded-xl border border-indigo-100 bg-indigo-50/85 px-4 py-3 text-xs text-indigo-700 dark:border-indigo-400/25 dark:bg-indigo-500/12 dark:text-indigo-200">
            Demo mode: enter any email and password to continue.
          </motion.div>

          <motion.div variants={inputContainerVariants} className="space-y-6">
            <motion.label variants={itemVariants} className="block text-sm font-medium text-gray-700 dark:text-slate-200">
              Email
              <input
                type="text"
                placeholder="you@example.com"
                className={inputClassName}
                value={email}
                onChange={(event) => {
                  setEmail(event.target.value);
                  if (error) {
                    setError("");
                  }
                }}
              />
            </motion.label>

            <motion.label variants={itemVariants} className="block text-sm font-medium text-gray-700 dark:text-slate-200">
              Password
              <input
                type="password"
                placeholder="Enter your password"
                className={inputClassName}
                value={password}
                onChange={(event) => {
                  setPassword(event.target.value);
                  if (error) {
                    setError("");
                  }
                }}
              />
            </motion.label>
          </motion.div>

          <motion.div variants={itemVariants} className="mt-4 text-right">
            <button type="button" className="text-sm font-medium text-indigo-600 transition hover:text-indigo-500 dark:text-indigo-300 dark:hover:text-indigo-200">
              Forgot password?
            </button>
          </motion.div>

          {error && (
            <motion.p variants={itemVariants} className="mt-4 text-sm font-medium text-red-500">
              {error}
            </motion.p>
          )}

          <motion.button
            variants={itemVariants}
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.98 }}
            className="mt-7 w-full rounded-xl bg-gradient-to-r from-indigo-600 to-blue-600 py-3 font-semibold text-white shadow-lg transition-all duration-300 hover:scale-105 hover:shadow-indigo-500/30"
            onClick={handleLogin}
          >
            Login →
          </motion.button>

          <motion.p variants={itemVariants} className="mt-6 text-center text-sm text-gray-600 dark:text-slate-300">
            Don’t have an account?{" "}
            <button type="button" className="font-semibold text-indigo-600 transition hover:text-indigo-500 dark:text-indigo-300 dark:hover:text-indigo-200">
              Sign up
            </button>
          </motion.p>
        </motion.div>
      </section>
    </div>
  );
};

export default LoginPage;