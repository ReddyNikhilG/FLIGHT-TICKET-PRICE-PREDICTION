import React from 'react';
import { Link } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { ArrowRight, Plane } from 'lucide-react';
import { motion } from 'framer-motion';

export default function CTASection() {
  return (
    <section className="py-20 sm:py-28">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="relative overflow-hidden rounded-3xl bg-primary p-10 sm:p-16 text-center"
        >
          {/* Background image */}
          <img
            src="https://images.unsplash.com/photo-1569629743817-70d8db6c323b?w=1400&auto=format&fit=crop&q=80"
            alt="Airport terminal"
            className="absolute inset-0 w-full h-full object-cover opacity-15 rounded-3xl"
          />
          {/* Decoration */}
          <div className="absolute top-0 right-0 w-64 h-64 bg-white/5 rounded-full -translate-y-1/2 translate-x-1/2" />
          <div className="absolute bottom-0 left-0 w-48 h-48 bg-white/5 rounded-full translate-y-1/2 -translate-x-1/2" />
          <Plane className="absolute top-8 right-12 w-16 h-16 text-white/10 -rotate-45" />

          <div className="relative z-10">
            <h2 className="font-heading text-3xl sm:text-4xl font-bold text-primary-foreground tracking-tight">
              Ready to Predict Your Flight Price?
            </h2>
            <p className="mt-4 text-primary-foreground/80 text-lg max-w-xl mx-auto">
              Try our ML model with your flight details and get an instant price estimate.
            </p>
            <Link to="/predict">
              <Button
                size="lg"
                variant="secondary"
                className="mt-8 rounded-full px-8 h-12 text-base font-heading font-semibold gap-2 group"
              >
                Start Predicting
                <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
              </Button>
            </Link>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
