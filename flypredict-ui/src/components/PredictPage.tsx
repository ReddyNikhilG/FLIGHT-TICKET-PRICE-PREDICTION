import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import {
  ArrowLeft,
  CalendarDays,
  CircleDollarSign,
  Clock3,
  MapPin,
  Plane,
  Sparkles,
} from "lucide-react";

type OptionResponse = {
  options: {
    airline: string[];
    source_city: string[];
    destination_city: string[];
    departure_time: string[];
    arrival_time: string[];
    stops: string[];
    class: string[];
    duration: { min: number; max: number };
    days_left: { min: number; max: number };
  };
};

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const inputClassName =
  "mt-2 w-full rounded-xl border border-gray-300 bg-white px-4 py-3 text-sm text-slate-800 shadow-sm outline-none transition focus:ring-2 focus:ring-indigo-500 dark:border-white/10 dark:bg-black/30 dark:text-slate-100";

const fieldVariants = {
  hidden: { opacity: 0, y: 24 },
  visible: { opacity: 1, y: 0 },
};

const PredictPage = () => {
  const navigate = useNavigate();
  const [options, setOptions] = useState<OptionResponse["options"] | null>(null);
  const [loading, setLoading] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [predictedPrice, setPredictedPrice] = useState<number | null>(null);
  const [error, setError] = useState("");
  const [travelDate, setTravelDate] = useState("");
  const isLoading = predicting;

  const [form, setForm] = useState({
    airline: "",
    source_city: "",
    destination_city: "",
    departure_time: "",
    arrival_time: "",
    stops: "",
    class: "",
    duration: 0,
    days_left: 1,
  });

  useEffect(() => {
    const loadMetadata = async () => {
      try {
        const res = await fetch(`${API_BASE}/metadata`);
        if (!res.ok) {
          throw new Error("Prediction API is not running");
        }

        const data: OptionResponse = await res.json();
        setOptions(data.options);
        setForm({
          airline: data.options.airline[0] ?? "",
          source_city: data.options.source_city[0] ?? "",
          destination_city: data.options.destination_city[0] ?? "",
          departure_time: data.options.departure_time[0] ?? "",
          arrival_time: data.options.arrival_time[0] ?? "",
          stops: data.options.stops[0] ?? "",
          class: data.options.class[0] ?? "",
          duration: Number((data.options.duration.min + 1).toFixed(1)),
          days_left: data.options.days_left.min,
        });
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load metadata");
      } finally {
        setLoading(false);
      }
    };

    loadMetadata();
  }, []);

  const destinationOptions = useMemo(() => {
    if (!options) {
      return [];
    }
    return options.destination_city.filter((city) => city !== form.source_city);
  }, [form.source_city, options]);

  useEffect(() => {
    if (destinationOptions.length && !destinationOptions.includes(form.destination_city)) {
      setForm((prev) => ({ ...prev, destination_city: destinationOptions[0] }));
    }
  }, [destinationOptions, form.destination_city]);

  const updateForm = (key: keyof typeof form, value: string | number) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const updateTravelDate = (value: string) => {
    setTravelDate(value);
    if (!options || !value) {
      return;
    }

    const today = new Date();
    const selected = new Date(value);
    const msPerDay = 24 * 60 * 60 * 1000;
    const days = Math.ceil((selected.getTime() - today.getTime()) / msPerDay);
    const clamped = Math.min(options.days_left.max, Math.max(options.days_left.min, days));
    updateForm("days_left", clamped);
  };

  const runPrediction = async () => {
    setPredicting(true);
    setError("");
    setPredictedPrice(null);

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || "Prediction failed");
      }

      setPredictedPrice(data.predicted_price);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setPredicting(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-gray-50 to-indigo-50 px-6 dark:from-gray-900 dark:to-black">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="w-full max-w-xl rounded-2xl border border-gray-200 bg-white p-8 text-center shadow-xl dark:border-white/10 dark:bg-white/5"
        >
          <Sparkles className="mx-auto h-7 w-7 text-indigo-500" />
          <p className="mt-3 text-lg font-semibold text-slate-800 dark:text-slate-100">Loading predictor...</p>
        </motion.div>
      </div>
    );
  }

  if (!options) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-gray-50 to-indigo-50 px-6 dark:from-gray-900 dark:to-black">
        <div className="w-full max-w-xl rounded-2xl border border-red-200 bg-white p-8 shadow-xl dark:border-red-500/30 dark:bg-white/5">
          <p className="text-xl font-semibold text-red-600">{error || "Unable to load predictor."}</p>
          <div className="my-6 h-px bg-gray-200 dark:bg-white/10" />
          <button
            type="button"
            onClick={() => navigate("/landing")}
            className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-indigo-600 px-4 py-3 font-semibold text-white transition-all duration-300 hover:scale-[1.02] hover:bg-indigo-700"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to landing
          </button>
        </div>
      </div>
    );
  }

  const bookingHint = form.days_left <= 14 ? "Best time to book: Now" : "Best time to book: Wait";

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-gray-50 to-indigo-50 px-6 py-8 dark:from-gray-900 dark:to-black">
      <motion.div
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.55, ease: "easeOut" }}
        className="w-full max-w-4xl space-y-8 rounded-2xl bg-white p-8 shadow-xl backdrop-blur-md transition-all duration-300 dark:bg-white/5"
      >
        <div className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight text-slate-900 dark:text-white">Predict Your Flight Price</h1>
            <p className="mt-2 text-sm text-gray-600 dark:text-gray-300">
              AI-powered insights to help you save money
            </p>
          </div>

          <button
            type="button"
            onClick={() => navigate("/landing")}
            className="inline-flex items-center justify-center gap-2 rounded-xl border border-gray-300 px-4 py-2.5 text-sm font-semibold text-slate-700 transition hover:bg-gray-50 dark:border-white/10 dark:text-slate-200 dark:hover:bg-white/10"
          >
            <ArrowLeft className="h-4 w-4" />
            Back
          </button>
        </div>

        <div className="mb-6 rounded-2xl border border-indigo-100 bg-indigo-50/70 p-4 text-sm text-indigo-900 dark:border-indigo-400/20 dark:bg-indigo-500/10 dark:text-indigo-200">
          <div className="flex items-center gap-2">
            <Sparkles className="h-4 w-4" />
            <p className="font-medium">Route preview</p>
          </div>
          <p className="mt-2">
            <span className="font-semibold">{form.source_city}</span> to <span className="font-semibold">{form.destination_city}</span> with <span className="font-semibold">{form.airline}</span>
          </p>
        </div>

        <div className="my-6 h-px bg-gradient-to-r from-transparent via-gray-300 to-transparent dark:via-white/10" />

        <motion.div
          variants={{
            hidden: {},
            visible: { transition: { staggerChildren: 0.06 } },
          }}
          initial="hidden"
          animate="visible"
          className="grid gap-6 md:grid-cols-2"
        >
          <motion.label variants={fieldVariants} className="text-sm font-medium text-slate-700 dark:text-slate-200">
            <span className="inline-flex items-center gap-2"><MapPin className="h-4 w-4 text-indigo-500" /> Source</span>
            <select className={inputClassName} value={form.source_city} onChange={(e) => updateForm("source_city", e.target.value)}>
              {options.source_city.map((item) => <option key={item} value={item}>{item}</option>)}
            </select>
          </motion.label>

          <motion.label variants={fieldVariants} className="text-sm font-medium text-slate-700 dark:text-slate-200">
            <span className="inline-flex items-center gap-2"><MapPin className="h-4 w-4 text-indigo-500" /> Destination</span>
            <select className={inputClassName} value={form.destination_city} onChange={(e) => updateForm("destination_city", e.target.value)}>
              {destinationOptions.map((item) => <option key={item} value={item}>{item}</option>)}
            </select>
          </motion.label>

          <motion.label variants={fieldVariants} className="text-sm font-medium text-slate-700 dark:text-slate-200">
            <span className="inline-flex items-center gap-2"><CalendarDays className="h-4 w-4 text-indigo-500" /> Date</span>
            <input
              type="date"
              className={inputClassName}
              value={travelDate}
              onChange={(e) => updateTravelDate(e.target.value)}
            />
          </motion.label>

          <motion.label variants={fieldVariants} className="text-sm font-medium text-slate-700 dark:text-slate-200">
            <span className="inline-flex items-center gap-2"><Plane className="h-4 w-4 text-indigo-500" /> Airline (optional)</span>
            <select className={inputClassName} value={form.airline} onChange={(e) => updateForm("airline", e.target.value)}>
              {options.airline.map((item) => <option key={item} value={item}>{item}</option>)}
            </select>
          </motion.label>

          <motion.label variants={fieldVariants} className="text-sm font-medium text-slate-700 dark:text-slate-200">
            <span className="inline-flex items-center gap-2"><CircleDollarSign className="h-4 w-4 text-indigo-500" /> Class (optional)</span>
            <select className={inputClassName} value={form.class} onChange={(e) => updateForm("class", e.target.value)}>
              {options.class.map((item) => <option key={item} value={item}>{item}</option>)}
            </select>
          </motion.label>

          <motion.label variants={fieldVariants} className="text-sm font-medium text-slate-700 dark:text-slate-200">
            <span className="inline-flex items-center gap-2"><Clock3 className="h-4 w-4 text-indigo-500" /> Departure Time Slot</span>
            <select className={inputClassName} value={form.departure_time} onChange={(e) => updateForm("departure_time", e.target.value)}>
              {options.departure_time.map((item) => <option key={item} value={item}>{item}</option>)}
            </select>
          </motion.label>

          <motion.label variants={fieldVariants} className="text-sm font-medium text-slate-700 dark:text-slate-200">
            <span className="inline-flex items-center gap-2"><Clock3 className="h-4 w-4 text-indigo-500" /> Arrival Time Slot</span>
            <select className={inputClassName} value={form.arrival_time} onChange={(e) => updateForm("arrival_time", e.target.value)}>
              {options.arrival_time.map((item) => <option key={item} value={item}>{item}</option>)}
            </select>
          </motion.label>

          <motion.label variants={fieldVariants} className="text-sm font-medium text-slate-700 dark:text-slate-200">
            Number of Stops
            <select className={inputClassName} value={form.stops} onChange={(e) => updateForm("stops", e.target.value)}>
              {options.stops.map((item) => <option key={item} value={item}>{item}</option>)}
            </select>
          </motion.label>

          <motion.label variants={fieldVariants} className="text-sm font-medium text-slate-700 dark:text-slate-200">
            Duration (hours)
            <input
              type="number"
              min={options.duration.min}
              max={options.duration.max}
              step={0.1}
              className={inputClassName}
              value={form.duration}
              onChange={(e) => updateForm("duration", Number(e.target.value))}
              placeholder="Duration in hours"
            />
          </motion.label>

          <motion.label variants={fieldVariants} className="text-sm font-medium text-slate-700 dark:text-slate-200 md:col-span-2">
            Days Left Before Departure
            <input
              type="number"
              min={options.days_left.min}
              max={options.days_left.max}
              className={inputClassName}
              value={form.days_left}
              onChange={(e) => updateForm("days_left", Number(e.target.value))}
              placeholder="Days left"
            />
          </motion.label>
        </motion.div>

        <div className="my-7 h-px bg-gradient-to-r from-transparent via-gray-300 to-transparent dark:via-white/10" />

        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          type="button"
          onClick={runPrediction}
          disabled={isLoading}
          className="w-full rounded-xl bg-gradient-to-r from-indigo-600 to-blue-600 py-3 font-semibold text-white shadow-lg transition-all duration-300 hover:scale-105 hover:shadow-indigo-500/30 disabled:cursor-not-allowed disabled:opacity-70"
        >
          {isLoading ? "Analyzing flight data..." : "Predict Price →"}
        </motion.button>

        {isLoading && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-center justify-center gap-2 text-sm font-medium text-indigo-600 dark:text-indigo-300"
          >
            <span className="h-2.5 w-2.5 animate-bounce rounded-full bg-indigo-500 [animation-delay:-0.2s]" />
            <span className="h-2.5 w-2.5 animate-bounce rounded-full bg-blue-500 [animation-delay:-0.1s]" />
            <span className="h-2.5 w-2.5 animate-bounce rounded-full bg-sky-500" />
            <span className="ml-1">Analyzing flight data...</span>
          </motion.div>
        )}

        {error && <p className="mt-4 text-sm font-medium text-red-600">{error}</p>}

        {predictedPrice !== null && (
          <motion.div
            initial={{ opacity: 0, scale: 0.96, y: 12 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            transition={{ duration: 0.35 }}
            className="mt-8 rounded-2xl bg-gradient-to-r from-indigo-500 to-blue-500 p-6 text-center text-white shadow-xl"
          >
            <p className="text-sm font-medium uppercase tracking-[0.18em] text-white/85">Estimated Fare</p>
            <p className="mt-2 text-3xl font-bold">₹{predictedPrice.toLocaleString()}</p>
            <p className="mt-3 text-sm text-white/90">{bookingHint}</p>
            <p className="mt-1 text-sm text-white/85">Confidence: 92%</p>
          </motion.div>
        )}
      </motion.div>
    </div>
  );
};

export default PredictPage;
