# frozen_string_literal: true

require "shoulder_segmenter"

RSpec.configure do |config|
  config.expect_with(:rspec) { |c| c.syntax = :expect }
  config.disable_monkey_patching!
  config.example_status_persistence_file_path = ".rspec_status"
end

module SpecPaths
  ROOT     = File.expand_path("..", __dir__)
  SCRIPT   = File.join(ROOT, "script")
  GOLDEN   = File.join(SCRIPT, "golden")
  FIXTURES = File.join(ROOT, "spec", "fixtures")

  def golden(name)
    File.join(GOLDEN, name)
  end

  def fixture(name)
    File.join(FIXTURES, name)
  end

  def model_weights_path
    ENV["SHOULDER_SEGMENTER_MODEL"] || File.join(SCRIPT, "totalsegmentator_part4_muscles.pt")
  end

  def golden_artifacts_available?
    File.exist?(golden("network_config.yaml")) &&
      File.exist?(golden("nnunet_architecture.yaml")) &&
      File.exist?(golden("sample_patch_in.bin")) &&
      File.exist?(golden("sample_patch_out.bin")) &&
      File.exist?(model_weights_path)
  end
end

RSpec.configure { |c| c.include SpecPaths }
