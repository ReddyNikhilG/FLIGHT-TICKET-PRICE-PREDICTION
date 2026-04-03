import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import { useAuth } from "../context/AuthContext";
import flight2Img from "../assets/flight2.jpg";

const heroImage = flight2Img;

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

type FormState = {
  airline: string;
  source_city: string;
  destination_city: string;
  departure_time: string;
  arrival_time: string;
  stops: string;
  class: string;
  duration: number;
  days_left: number;
};

const features = [
  {
    title: "AI Fare Intelligence",
    desc: "Premium ML pricing engine tuned on your dataset for route-level ticket estimation.",
    stat: "95%",
  },
  {
    title: "Best Booking Window",
    desc: "Get timing confidence with fast predictions before committing to ticket purchases.",
    stat: "40%",
  },
  {
    title: "Multi-Airline Scope",
    desc: "Compare trends across airline and class combinations from one polished interface.",
    stat: "200+",
  },
];

const faqs = [
  {
    q: "How accurate are the price predictions?",
    a: "Our AI models achieve up to 95% accuracy by analyzing historical pricing, demand trends, and airline data.",
  },
  {
    q: "When is the best time to book a flight?",
    a: "Prices are typically lowest 4–8 weeks before departure for domestic flights and 2–4 months for international flights.",
  },
  {
    q: "Does FlyPredict support all airlines?",
    a: "Yes, we analyze data from 200+ airlines worldwide.",
  },
  {
    q: "Is this service free to use?",
    a: "Yes, FlyPredict offers free predictions with optional premium features coming soon.",
  },
  {
    q: "Can I track price changes?",
    a: "Yes, you can monitor trends and identify the best time to book.",
  },
  {
    q: "How does the AI predict prices?",
    a: "It uses machine learning trained on millions of flight records and seasonal trends.",
  },
];

const testimonials = [
  {
    name: "Aarav Mehta",
    role: "Travel Ops Lead",
    quote: "The interface feels premium and the predictions are fast enough to use in real workflows.",
  },
  {
    name: "Sara Khan",
    role: "Product Manager",
    quote: "The route overview and polished motion make the experience feel like a real SaaS product.",
  },
  {
    name: "Nikhil Sharma",
    role: "Founder",
    quote: "This is the kind of clean, dependable UI that builds trust before the first prediction even runs.",
  },
];

const howItWorks = [
  {
    step: "01",
    title: "Enter route details",
    desc: "Choose airline, source, destination, cabin class, and trip timing from the premium form.",
  },
  {
    step: "02",
    title: "Predict instantly",
    desc: "Your trained model returns the price with a clear, centered result card and smooth reveal.",
  },
  {
    step: "03",
    title: "Review the route",
    desc: "The map resolves both cities, plots the path, and fits bounds so the trip is easy to inspect.",
  },
];

const navItems = ["Home", "Features", "How It Works", "Testimonials", "Predict", "FAQ", "Contact"];

const heroAvatars = ["A", "S", "N", "K"];

function SectionHeading(props: { title: string; subtitle: string }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.2 }}
      transition={{ duration: 0.5 }}
      className="mx-auto max-w-3xl text-center"
    >
      <h2 className="text-3xl font-black tracking-tight text-slate-900 dark:text-white md:text-5xl">
        {props.title}
      </h2>
      <p className="mx-auto mt-4 max-w-2xl text-base leading-relaxed text-slate-600 dark:text-slate-300 md:text-lg">
        {props.subtitle}
      </p>
    </motion.div>
  );
}

function HeroAnalyticsCard() {
  return (
    <motion.div
      animate={{ y: [0, -12, 0] }}
      transition={{ duration: 6.5, repeat: Infinity, ease: "easeInOut" }}
      className="relative mx-auto w-full max-w-md rounded-2xl border border-white/15 bg-white/10 p-4 shadow-2xl shadow-black/25 backdrop-blur-md"
    >
      <div className="rounded-2xl border border-white/10 bg-slate-950/40 p-5 text-white">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-xs uppercase tracking-[0.24em] text-white/60">Price Trend</p>
            <h3 className="mt-2 text-xl font-bold tracking-tight">Price Trend — NYC → LON</h3>
          </div>
          <div className="rounded-full border border-emerald-400/20 bg-emerald-500/15 px-3 py-1 text-xs font-semibold text-emerald-300">
            Live insight
          </div>
        </div>

        <div className="mt-6 rounded-2xl border border-white/10 bg-white/5 p-4">
          <svg viewBox="0 0 320 150" className="h-36 w-full">
            <defs>
              <linearGradient id="heroLine" x1="0" x2="1" y1="0" y2="0">
                <stop offset="0%" stopColor="#60a5fa" />
                <stop offset="100%" stopColor="#6366f1" />
              </linearGradient>
              <linearGradient id="heroFill" x1="0" x2="0" y1="0" y2="1">
                <stop offset="0%" stopColor="rgba(99,102,241,0.45)" />
                <stop offset="100%" stopColor="rgba(99,102,241,0)" />
              </linearGradient>
            </defs>
            <path
              d="M20 114 C48 104, 64 96, 88 88 C112 80, 128 92, 150 74 C173 56, 192 62, 213 52 C234 42, 253 44, 276 36 C292 31, 304 29, 310 27"
              fill="none"
              stroke="url(#heroLine)"
              strokeWidth="5"
              strokeLinecap="round"
            />
            <path
              d="M20 114 C48 104, 64 96, 88 88 C112 80, 128 92, 150 74 C173 56, 192 62, 213 52 C234 42, 253 44, 276 36 C292 31, 304 29, 310 27 L310 146 L20 146 Z"
              fill="url(#heroFill)"
            />
            <circle cx="150" cy="74" r="6" fill="#fff" />
            <circle cx="276" cy="36" r="6" fill="#fff" />
          </svg>
        </div>

        <div className="mt-5 grid grid-cols-2 gap-3 text-sm">
          <div className="rounded-xl border border-white/10 bg-white/5 p-3">
            <p className="text-white/60">Best price</p>
            <p className="mt-1 text-2xl font-black tracking-tight">$95</p>
          </div>
          <div className="rounded-xl border border-white/10 bg-white/5 p-3">
            <p className="text-white/60">Savings</p>
            <p className="mt-1 text-2xl font-black tracking-tight text-emerald-300">-47%</p>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

function HeroSection(props: {
  onLogout: () => void;
  theme: "light" | "dark";
  onToggleTheme: () => void;
  onStartPrediction: () => void;
}) {
  const navigate = useNavigate();
  const [activeSection, setActiveSection] = useState("Home");

  useEffect(() => {
    const sectionMap: Array<{ id: string; label: string }> = [
      { id: "home", label: "Home" },
      { id: "features", label: "Features" },
      { id: "how", label: "How It Works" },
      { id: "testimonials", label: "Testimonials" },
      { id: "faq", label: "FAQ" },
      { id: "contact", label: "Contact" },
    ];

    const updateActiveSection = () => {
      const scrollPosition = window.scrollY + 160;
      let current = "Home";

      for (const section of sectionMap) {
        const element = document.getElementById(section.id);
        if (!element) {
          continue;
        }
        if (scrollPosition >= element.offsetTop) {
          current = section.label;
        }
      }

      setActiveSection(current);
    };

    updateActiveSection();
    window.addEventListener("scroll", updateActiveSection, { passive: true });
    return () => window.removeEventListener("scroll", updateActiveSection);
  }, []);

  return (
    <section
      id="home"
      className="hero-cinematic relative h-screen overflow-hidden bg-cover bg-center"
      style={{ backgroundImage: `url(${heroImage})` }}
    >
      <div className="absolute inset-0 bg-gradient-to-b from-black/75 via-black/55 to-black/20" />
      <div className="absolute inset-0 bg-black/20" />

      <header className="fixed left-0 right-0 top-0 z-50 w-full border-b border-white/15 bg-white/70 backdrop-blur-md dark:bg-black/40">
        <div className="mx-auto flex h-20 w-full max-w-7xl items-center justify-between px-6">
          <div className="flex items-center gap-3 text-slate-900 dark:text-white">
            <span className="logo-badge">✈</span>
            <div>
              <p className="text-xs uppercase tracking-[0.28em] text-slate-500 dark:text-white/60">FlyPredict</p>
              <p className="text-base font-extrabold tracking-tight md:text-lg">Flight Intelligence</p>
            </div>
          </div>

          <nav className="hidden items-center gap-7 md:flex">
            {navItems.map((label) => (
              <button
                key={label}
                type="button"
                onClick={() => {
                  if (label === "Predict") {
                    navigate("/predict");
                    return;
                  }

                  if (label === "Home") {
                    window.scrollTo({ top: 0, behavior: "smooth" });
                    return;
                  }

                  if (label === "How It Works") {
                    document.getElementById("how")?.scrollIntoView({ behavior: "smooth", block: "start" });
                    return;
                  }

                  const sectionId = label.toLowerCase().replace(/\s+/g, "-");
                  document.getElementById(sectionId)?.scrollIntoView({ behavior: "smooth", block: "start" });
                }}
                className={`group relative text-sm font-medium transition ${
                  activeSection === label
                    ? "text-slate-900 dark:text-white"
                    : "text-slate-700 hover:text-slate-900 dark:text-white/80 dark:hover:text-white"
                }`}
              >
                {label}
                <span
                  className={`absolute -bottom-2 left-0 h-px bg-slate-900 transition-all duration-300 dark:bg-white ${
                    activeSection === label ? "w-full" : "w-0 group-hover:w-full"
                  }`}
                />
              </button>
            ))}
          </nav>

          <div className="flex items-center gap-2">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.98 }}
              onClick={props.onToggleTheme}
              className="rounded-xl border border-slate-200 bg-white/80 px-4 py-2 text-sm font-semibold text-slate-800 shadow-sm backdrop-blur-md transition hover:bg-white dark:border-white/10 dark:bg-white/10 dark:text-white"
            >
              {props.theme === "dark" ? "☀️ Light" : "🌙 Dark"}
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.98 }}
              onClick={props.onLogout}
              className="rounded-xl border border-slate-200 bg-white/80 px-4 py-2 text-sm font-semibold text-slate-800 shadow-sm backdrop-blur-md transition hover:bg-white dark:border-white/10 dark:bg-white/10 dark:text-white"
            >
              Logout
            </motion.button>
          </div>
        </div>
      </header>

      <div className="relative z-10 flex h-full items-center px-6 pt-10 md:pt-12">
        <div className="mx-auto grid w-full max-w-7xl items-center gap-10 md:grid-cols-2">
          <motion.div
            initial={{ opacity: 0, y: 34 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.75, ease: "easeOut" }}
            className="max-w-2xl text-center md:text-left"
          >
            <div className="mx-auto flex w-fit items-center rounded-full border border-white/15 bg-white/10 px-4 py-1.5 text-sm font-medium text-white backdrop-blur-md md:mx-0">
              Premium fare analytics
            </div>
            <h1 className="display-font mt-5 text-5xl font-extrabold leading-tight text-white [text-shadow:0_10px_28px_rgba(0,0,0,0.35)] md:text-6xl lg:text-7xl">
              Predict flight prices before
              <span className="block bg-gradient-to-r from-blue-400 to-indigo-500 bg-clip-text text-transparent">
                you book
              </span>
            </h1>
            <p className="mx-auto mt-6 max-w-xl text-base leading-relaxed text-slate-100 md:mx-0 md:text-lg">
              Use AI-powered fare predictions to spot savings, compare timing, and book with more confidence before prices move.
            </p>
            <div className="mt-8 flex flex-col gap-3 sm:flex-row sm:justify-center md:justify-start">
              <motion.button
                whileHover={{ scale: 1.05, boxShadow: "0 18px 45px rgba(79,70,229,0.35)" }}
                whileTap={{ scale: 0.98 }}
                onClick={props.onStartPrediction}
                className="w-full rounded-xl bg-gradient-to-r from-indigo-600 to-blue-600 px-6 py-3 text-sm font-semibold text-white shadow-lg transition-all duration-300 hover:scale-105 hover:shadow-indigo-500/30 sm:w-auto"
              >
                Start Prediction →
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => document.getElementById("features")?.scrollIntoView({ behavior: "smooth", block: "start" })}
                className="w-full rounded-xl border border-white/30 px-6 py-3 text-sm font-semibold text-white transition hover:bg-white/10 sm:w-auto"
              >
                Learn More
              </motion.button>
            </div>

            <div className="mt-8 flex items-center justify-center gap-4 md:justify-start">
              <div className="flex -space-x-3">
                {heroAvatars.map((avatar, index) => (
                  <div
                    key={avatar}
                    className="flex h-10 w-10 items-center justify-center rounded-full border-2 border-white/15 bg-white/20 text-sm font-bold text-white shadow-lg backdrop-blur-sm"
                    style={{ backgroundColor: `hsla(${210 + index * 15}, 90%, 70%, 0.18)` }}
                  >
                    {avatar}
                  </div>
                ))}
              </div>
              <p className="text-sm font-medium text-white/80 md:text-base">
                12,000+ travelers trust FlyPredict
              </p>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.75, ease: "easeOut", delay: 0.15 }}
            className="flex justify-center md:justify-end"
          >
            <HeroAnalyticsCard />
          </motion.div>
        </div>
      </div>
    </section>
  );
}

function FeaturesSection() {
  return (
    <section id="features" className="px-6 py-20">
      <div className="mx-auto max-w-7xl">
        <SectionHeading title="Built for premium booking decisions" subtitle="High-end visual system, strong hierarchy, and practical model outputs." />

        <div className="mt-10 grid gap-6 md:grid-cols-3">
          {features.map((feature) => (
            <motion.article
              key={feature.title}
              initial={{ opacity: 0, y: 18 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.2 }}
              whileHover={{ y: -8, scale: 1.05 }}
              transition={{ duration: 0.3, ease: "easeOut" }}
              className="rounded-2xl border border-slate-200 bg-white p-7 shadow-md transition-all duration-300 hover:shadow-xl dark:border-white/10 dark:bg-white/5 dark:backdrop-blur-md"
            >
              <h3 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-white">{feature.title}</h3>
              <p className="mt-3 text-slate-600 dark:text-slate-300">{feature.desc}</p>
              <p className="mt-5 text-3xl font-black text-sky-600 dark:text-sky-400">{feature.stat}</p>
            </motion.article>
          ))}
        </div>
      </div>
    </section>
  );
}

function HowItWorksSection() {
  return (
    <section id="how" className="px-6 py-20">
      <div className="mx-auto max-w-7xl">
        <SectionHeading title="How It Works" subtitle="A simple, premium flow from route input to prediction and route visualization." />

        <div className="relative mt-12 grid gap-6 lg:grid-cols-3 lg:gap-8">
          <div className="absolute left-1/2 top-12 hidden h-[2px] w-[70%] -translate-x-1/2 bg-gradient-to-r from-blue-500 via-indigo-500 to-blue-500 lg:block" />
          {howItWorks.map((step, index) => (
            <motion.article
              key={step.step}
              initial={{ opacity: 0, y: 22 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.2 }}
              transition={{ duration: 0.4, delay: index * 0.08 }}
              className="relative rounded-2xl border border-slate-200 bg-white p-7 shadow-md transition-all duration-300 hover:scale-[1.03] hover:shadow-xl dark:border-white/10 dark:bg-white/5 dark:backdrop-blur-md"
            >
              <div className="mb-5 flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-500 to-indigo-600 text-sm font-black text-white shadow-lg shadow-indigo-500/30">
                {step.step}
              </div>
              <h3 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-white">{step.title}</h3>
              <p className="mt-3 text-slate-600 dark:text-slate-300">{step.desc}</p>
            </motion.article>
          ))}
        </div>
      </div>
    </section>
  );
}

function TestimonialsSection() {
  return (
    <section id="testimonials" className="px-6 py-20">
      <div className="mx-auto max-w-7xl">
        <SectionHeading title="Testimonials" subtitle="A few words from people who value speed, polish, and clarity." />

        <div className="mt-10 grid gap-6 md:grid-cols-3">
          {testimonials.map((testimonial, index) => (
            <motion.article
              key={testimonial.name}
              initial={{ opacity: 0, y: 18 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.2 }}
              transition={{ duration: 0.4, delay: index * 0.08 }}
              whileHover={{ scale: 1.05 }}
              className="rounded-2xl border border-slate-200 bg-white p-7 shadow-md transition-all duration-300 hover:shadow-xl dark:border-white/10 dark:bg-white/5 dark:backdrop-blur-md"
            >
              <p className="text-lg leading-relaxed text-slate-700 dark:text-slate-200">“{testimonial.quote}”</p>
              <div className="mt-6">
                <p className="font-bold text-slate-900 dark:text-white">{testimonial.name}</p>
                <p className="text-sm text-slate-500 dark:text-slate-400">{testimonial.role}</p>
              </div>
            </motion.article>
          ))}
        </div>
      </div>
    </section>
  );
}

function CTASection() {
  return (
    <section id="cta" className="px-6 py-20">
      <div className="mx-auto max-w-7xl">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.2 }}
          transition={{ duration: 0.5 }}
          className="rounded-3xl bg-gradient-to-r from-indigo-600 to-blue-600 p-10 text-white shadow-xl shadow-indigo-600/30 dark:shadow-[0_30px_80px_rgba(59,130,246,0.28)]"
        >
          <div className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr] lg:items-center">
            <div>
              <h3 className="text-3xl font-black tracking-tight md:text-5xl">Ready to optimize your next booking?</h3>
              <p className="mt-4 max-w-2xl text-base leading-relaxed text-white/85 md:text-lg">
                Use the predictor above or contact us for enterprise model integration.
              </p>
            </div>
            <div className="flex flex-wrap gap-3 lg:justify-end">
              <button
                type="button"
                onClick={() => document.getElementById("predict")?.scrollIntoView({ behavior: "smooth", block: "start" })}
                className="rounded-xl bg-white px-5 py-3 text-sm font-bold text-indigo-700 shadow-xl shadow-black/10 transition hover:scale-105"
              >
                Start Prediction
              </button>
              <button
                type="button"
                onClick={() => document.getElementById("contact-form")?.scrollIntoView({ behavior: "smooth", block: "start" })}
                className="rounded-xl border border-white/25 bg-white/10 px-5 py-3 text-sm font-bold text-white backdrop-blur-md transition hover:scale-105 hover:bg-white/15"
              >
                Contact Us
              </button>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}

function FaqSection() {
  const [openFaqIndex, setOpenFaqIndex] = useState<number | null>(0);

  return (
    <section id="faq" className="px-6 py-20">
      <div className="mx-auto max-w-4xl">
        <SectionHeading title="Frequently Asked Questions" subtitle="Answers to the most common questions about the prediction flow and model usage." />

        <div className="mt-8 space-y-4">
          {faqs.map((faq, idx) => {
            const isOpen = openFaqIndex === idx;
            return (
              <motion.article
                key={faq.q}
                layout
                whileHover={{ y: -2 }}
                className="rounded-2xl border border-slate-200 bg-white p-2 shadow-sm transition-all duration-300 hover:shadow-md dark:border-white/10 dark:bg-white/5 dark:backdrop-blur-md"
              >
                <button
                  type="button"
                  onClick={() => setOpenFaqIndex(isOpen ? null : idx)}
                  className="flex w-full items-center justify-between px-4 py-4 text-left transition hover:bg-slate-50 dark:hover:bg-white/5"
                >
                  <span className="font-semibold text-slate-900 dark:text-white">{faq.q}</span>
                  <motion.span animate={{ rotate: isOpen ? 180 : 0 }} transition={{ duration: 0.3 }}>
                    ⌄
                  </motion.span>
                </button>
                <AnimatePresence initial={false}>
                  {isOpen && (
                    <motion.p
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{ opacity: 0, height: 0 }}
                      transition={{ duration: 0.28 }}
                      className="overflow-hidden px-4 pb-4 text-slate-600 dark:text-slate-300"
                    >
                      {faq.a}
                    </motion.p>
                  )}
                </AnimatePresence>
              </motion.article>
            );
          })}
        </div>
      </div>
    </section>
  );
}

function ContactSection() {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [message, setMessage] = useState("");
  const [success, setSuccess] = useState("");
  const [error, setError] = useState("");

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSuccess("");
    setError("");

    if (!name.trim() || !email.trim() || !message.trim()) {
      setError("Please fill in all required fields.");
      return;
    }

    setSuccess("Thanks for reaching out. We will contact you shortly.");
    setName("");
    setEmail("");
    setMessage("");
  };

  return (
    <section id="contact" className="px-6 pb-20 pt-8">
      <div className="mx-auto max-w-7xl">
        <motion.div
          initial={{ opacity: 0.85 }}
          animate={{ opacity: [0.85, 1, 0.85] }}
          transition={{ duration: 2.3, ease: "easeInOut", repeat: Infinity }}
          className="rounded-3xl border border-slate-200 bg-white p-10 shadow-lg dark:border-white/10 dark:bg-white/5 dark:backdrop-blur-md"
        >
          <div id="contact-form" className="grid gap-6 lg:grid-cols-2 lg:items-start">
            <div>
              <h3 className="text-3xl font-black tracking-tight text-slate-900 dark:text-white">
                Ready to optimize your next booking?
              </h3>
              <p className="mt-4 text-slate-600 dark:text-slate-300">
                Use the predictor above or contact us for enterprise model integration.
              </p>
            </div>

            <form className="grid gap-4" onSubmit={handleSubmit}>
              <input
                type="text"
                placeholder="Your name"
                required
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="rounded-xl border border-slate-200 bg-white px-4 py-3 text-slate-900 outline-none transition-all duration-300 focus:ring-2 focus:ring-indigo-500 dark:border-white/10 dark:bg-white/5 dark:text-white"
              />
              <input
                type="email"
                placeholder="Work email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="rounded-xl border border-slate-200 bg-white px-4 py-3 text-slate-900 outline-none transition-all duration-300 focus:ring-2 focus:ring-indigo-500 dark:border-white/10 dark:bg-white/5 dark:text-white"
              />
              <textarea
                rows={4}
                placeholder="Tell us what you want to build"
                required
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                className="rounded-xl border border-slate-200 bg-white px-4 py-3 text-slate-900 outline-none transition-all duration-300 focus:ring-2 focus:ring-indigo-500 dark:border-white/10 dark:bg-white/5 dark:text-white"
              />
              <button
                type="submit"
                className="rounded-xl bg-slate-900 px-5 py-3 font-semibold text-white transition hover:scale-[1.02] hover:bg-slate-800 dark:bg-white dark:text-slate-900 dark:hover:bg-slate-100"
              >
                Send Message
              </button>
              {error && <p className="text-sm font-medium text-red-500">{error}</p>}
              {success && <p className="text-sm font-medium text-emerald-500">{success}</p>}
            </form>
          </div>
        </motion.div>
      </div>
    </section>
  );
}

function LandingPage() {
  const navigate = useNavigate();
  const { logout } = useAuth();
  const [theme, setTheme] = useState<"light" | "dark">("light");

  useEffect(() => {
    const savedTheme = localStorage.getItem("flypredict_theme");
    const nextTheme = savedTheme === "dark" ? "dark" : "light";
    setTheme(nextTheme);
    document.documentElement.classList.toggle("dark", nextTheme === "dark");
  }, []);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    localStorage.setItem("flypredict_theme", theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme((current) => (current === "dark" ? "light" : "dark"));
  };

  const handleLogout = () => {
    logout();
    navigate("/login", { replace: true });
  };

  return (
    <div className="min-h-screen bg-white text-slate-900 transition-colors duration-300 dark:bg-slate-950 dark:text-white">
      <HeroSection theme={theme} onToggleTheme={toggleTheme} onLogout={handleLogout} onStartPrediction={() => navigate("/predict")} />

      <FeaturesSection />
      <HowItWorksSection />
      <TestimonialsSection />
      <CTASection />
      <FaqSection />
      <ContactSection />
    </div>
  );
}

export default LandingPage;
