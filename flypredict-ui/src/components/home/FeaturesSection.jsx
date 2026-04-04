import React from 'react';
import { motion } from 'framer-motion';
import { Brain, TrendingDown, Clock, BarChart3, Shield, Sparkles } from 'lucide-react';

const features = [
  {
    icon: Brain,
    title: 'ML Powered',
    description: 'Random Forest algorithm trained on real flight data for high-accuracy predictions.',
    color: 'bg-primary/10 text-primary',
  },
  {
    icon: TrendingDown,
    title: 'Price Insights',
    description: 'Understand how class, duration, and booking time affect your flight price.',
    color: 'bg-accent/10 text-accent',
  },
  {
    icon: Clock,
    title: 'Instant Results',
    description: 'Get predictions in real-time - no waiting, no complex setup required.',
    color: 'bg-chart-3/10 text-chart-3',
  },
  {
    icon: BarChart3,
    title: 'Model Analytics',
    description: 'Explore detailed model performance, feature importance, and cross-validation scores.',
    color: 'bg-chart-4/10 text-chart-4',
  },
  {
    icon: Shield,
    title: 'Reliable Model',
    description: 'Cross-validated with 96.85% mean R2 and 0.13% standard deviation for consistency.',
    color: 'bg-chart-5/10 text-chart-5',
  },
  {
    icon: Sparkles,
    title: 'Smart Features',
    description: 'Engineered features capture airline, route, timing, and booking patterns.',
    color: 'bg-chart-2/10 text-chart-2',
  },
];

export default function FeaturesSection() {
  return (
    <section className="py-20 sm:py-28 bg-card/50 relative overflow-hidden">
      <img
        src="https://images.unsplash.com/photo-1464037866556-6812c9d1c72e?w=1600&auto=format&fit=crop&q=80"
        alt="Clouds from above"
        className="absolute inset-0 w-full h-full object-cover opacity-[0.04]"
      />
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center max-w-2xl mx-auto mb-16">
          <h2 className="font-heading text-3xl sm:text-4xl font-bold tracking-tight">
            Why FlyPredict?
          </h2>
          <p className="mt-4 text-muted-foreground text-lg">
            Built with a rigorous ML pipeline - from data cleaning to deployment.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, i) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4, delay: i * 0.08 }}
              className="group p-6 rounded-2xl border border-border/50 bg-card hover:shadow-lg hover:shadow-primary/5 transition-all duration-300"
            >
              <div className={`w-11 h-11 rounded-xl ${feature.color} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform`}>
                <feature.icon className="w-5 h-5" />
              </div>
              <h3 className="font-heading font-semibold text-lg mb-2">{feature.title}</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">{feature.description}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
