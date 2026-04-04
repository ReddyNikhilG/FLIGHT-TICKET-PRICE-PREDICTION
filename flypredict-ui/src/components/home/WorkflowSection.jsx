import React from 'react';
import { motion } from 'framer-motion';
import { Database, Search, Wrench, GraduationCap, TestTube, Rocket } from 'lucide-react';

const steps = [
  { icon: Database, label: 'Data Collection', desc: 'Gathered 22,000+ flight records' },
  { icon: Search, label: 'EDA & Cleaning', desc: 'Handled missing values & outliers' },
  { icon: Wrench, label: 'Feature Engineering', desc: 'Encoded categories & scaled features' },
  { icon: GraduationCap, label: 'Model Training', desc: 'Trained 3 regression algorithms' },
  { icon: TestTube, label: 'Evaluation', desc: 'Cross-validated for reliability' },
  { icon: Rocket, label: 'Deployment', desc: 'Flask API + React frontend' },
];

export default function WorkflowSection() {
  return (
    <section className="py-20 sm:py-28">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center max-w-2xl mx-auto mb-16">
          <h2 className="font-heading text-3xl sm:text-4xl font-bold tracking-tight">
            ML Pipeline
          </h2>
          <p className="mt-4 text-muted-foreground text-lg">
            End-to-end machine learning workflow from raw data to live predictions.
          </p>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
          {steps.map((step, i) => (
            <motion.div
              key={step.label}
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.3, delay: i * 0.1 }}
              className="relative flex flex-col items-center text-center p-5 rounded-2xl border border-border/50 bg-card hover:border-primary/30 transition-colors group"
            >
              <div className="absolute -top-3 -left-1 w-6 h-6 rounded-full bg-primary text-primary-foreground text-xs font-bold flex items-center justify-center">
                {i + 1}
              </div>
              <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center mb-3 group-hover:bg-primary/20 transition-colors">
                <step.icon className="w-5 h-5 text-primary" />
              </div>
              <h3 className="font-heading font-semibold text-sm mb-1">{step.label}</h3>
              <p className="text-xs text-muted-foreground">{step.desc}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
