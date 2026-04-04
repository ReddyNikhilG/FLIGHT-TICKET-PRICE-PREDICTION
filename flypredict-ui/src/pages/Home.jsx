import React from 'react';
import HeroSection from '@/components/home/HeroSection';
import FeaturesSection from '@/components/home/FeaturesSection';
import WorkflowSection from '@/components/home/WorkflowSection';
import TechStackSection from '@/components/home/TechStackSection';
import CTASection from '@/components/home/CTASection';

export default function Home() {
  return (
    <div className="w-full overflow-hidden">
      <HeroSection />
      <FeaturesSection />
      <WorkflowSection />
      <TechStackSection />
      <CTASection />
    </div>
  );
}
