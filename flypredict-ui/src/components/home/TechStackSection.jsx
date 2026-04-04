import React from 'react';
import { motion } from 'framer-motion';

const techStack = [
  { name: 'Python', category: 'Backend' },
  { name: 'Flask', category: 'Backend' },
  { name: 'Scikit-learn', category: 'ML' },
  { name: 'Pandas', category: 'ML' },
  { name: 'NumPy', category: 'ML' },
  { name: 'React', category: 'Frontend' },
  { name: 'Tailwind CSS', category: 'Frontend' },
  { name: 'Jupyter', category: 'Tools' },
];

const categoryColors = {
  Backend: 'border-primary/30 bg-primary/5 text-primary',
  ML: 'border-accent/30 bg-accent/5 text-accent',
  Frontend: 'border-chart-3/30 bg-chart-3/5 text-chart-3',
  Tools: 'border-chart-4/30 bg-chart-4/5 text-chart-4',
};

export default function TechStackSection() {
  return (
    <section className="py-20 sm:py-28 bg-card/50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center max-w-2xl mx-auto mb-12">
          <h2 className="font-heading text-3xl sm:text-4xl font-bold tracking-tight">
            Tech Stack
          </h2>
          <p className="mt-4 text-muted-foreground text-lg">
            Built with modern tools for machine learning and web development.
          </p>
        </div>

        <div className="flex flex-wrap justify-center gap-3 max-w-2xl mx-auto">
          {techStack.map((tech, i) => (
            <motion.div
              key={tech.name}
              initial={{ opacity: 0, scale: 0.8 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.3, delay: i * 0.05 }}
              className={`px-5 py-2.5 rounded-full border text-sm font-medium ${categoryColors[tech.category]}`}
            >
              {tech.name}
            </motion.div>
          ))}
        </div>

        {/* Legend */}
        <div className="flex flex-wrap justify-center gap-6 mt-8">
          {Object.keys(categoryColors).map((cat) => (
            <div key={cat} className="flex items-center gap-2 text-xs text-muted-foreground">
              <div className={`w-2.5 h-2.5 rounded-full ${categoryColors[cat].split(' ')[1]}`} />
              {cat}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
