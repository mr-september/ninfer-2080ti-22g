#include "options.h"
#include "product/defaults.h"

#include <filesystem>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

ninfer::cli::Options parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return ninfer::cli::parse_options(static_cast<int>(argv.size()), argv.data());
}

bool rejects(const std::function<void()>& operation) {
    try {
        operation();
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures      = 0;
    const auto choice = parse({"ninfer", "model.ninfer", "--prompt", "hello", "--choice", "",
                               "--choice", "a|b", "--choice", "你好"});
    failures += check(choice.choices == std::vector<std::string>{"", "a|b", "你好"},
                      "choice flags changed literal text or order");
    const auto defaults = parse({"ninfer", "model.ninfer", "--prompt", "hello"});
    failures += check(defaults.max_context == ninfer::product::kDefaultMaxContext &&
                          defaults.kv_capacity.explicit_tokens == defaults.max_context &&
                          defaults.kv_cache == ninfer::product::default_kv_cache(),
                      "CLI defaults do not follow the product defaults");
    failures += check(defaults.speculative.backend == ninfer::SpeculativeBackend::Mtp &&
                          defaults.speculative.draft_tokens == ninfer::product::kDefaultDraftTokens &&
                          defaults.speculative.proposal_head == ninfer::ProposalHead::Auto,
                      "MTP with the automatic proposal head is not the CLI default");
    failures += check(defaults.chat_template_path == NINFER_DEFAULT_CHAT_TEMPLATE &&
                          std::filesystem::exists(defaults.chat_template_path),
                      "the vendored Sharp template is not the CLI default");
    failures += check(parse({"ninfer", "model.ninfer", "--prompt", "hello", "--chat-template",
                             "x.jinja"})
                              .chat_template_path == "x.jinja",
                      "--chat-template did not override the default template");
    failures += check(parse({"ninfer", "model.ninfer", "--prompt", "hello",
                             "--artifact-chat-template"})
                          .chat_template_path.empty(),
                      "--artifact-chat-template did not select the artifact template");
    failures += check(rejects([] {
                          (void)parse({"ninfer", "model.ninfer", "--prompt", "hello",
                                       "--artifact-chat-template", "--chat-template", "x.jinja"});
                      }),
                      "--artifact-chat-template was accepted with --chat-template");
    const auto plain = parse({"ninfer", "model.ninfer", "--prompt", "hello", "--spec", "none"});
    failures += check(plain.speculative.backend == ninfer::SpeculativeBackend::None &&
                          plain.speculative.draft_tokens == 0 &&
                          plain.speculative.proposal_head == ninfer::ProposalHead::Full,
                      "--spec none did not clear the speculative defaults");
    failures += check(rejects([] {
                          (void)parse({"ninfer", "model.ninfer", "--prompt", "hello", "--spec",
                                       "none", "--draft-tokens", "3"});
                      }),
                      "--draft-tokens was accepted with --spec none");
    failures += check(rejects([] {
                          (void)parse({"ninfer", "model.ninfer", "--prompt", "hello", "--spec",
                                       "dflash"});
                      }),
                      "--spec dflash was accepted without its own --draft-tokens");
    failures +=
        check(parse({"ninfer", "model.ninfer", "--prompt", "hello", "--regex", ""}).regex == "",
              "empty regex must be distinct from no regex");
    failures += check(rejects([] {
                          (void)parse({"ninfer", "model.ninfer", "--prompt", "hello", "--regex", "",
                                       "--choice", "yes"});
                      }),
                      "regex and choice were accepted together");
    failures += check(rejects([] {
                          (void)parse({"ninfer", "model.ninfer", "--prompt", "hello", "--regex",
                                       "a", "--json-object"});
                      }),
                      "regex and JSON were accepted together");
    failures += check(
        parse({"ninfer", "model.ninfer", "--prompt", "hello", "--grammar-file", "answer.gbnf"})
                .grammar_path == "answer.gbnf",
        "grammar file argument was lost");
    failures +=
        check(parse({"ninfer", "model.ninfer", "--prompt", "hello", "--json-object"}).json_object,
              "JSON object CLI option lost");
    failures += check(
        parse({"ninfer", "model.ninfer", "--prompt", "hello", "--json-schema-file", "answer.json"})
                .json_schema_path == "answer.json",
        "schema file CLI option lost");
    failures += check(rejects([] {
                          (void)parse({"ninfer", "model.ninfer", "--prompt", "hello",
                                       "--json-object", "--grammar-file", "a.gbnf"});
                      }),
                      "conflicting CLI constraints accepted");
    const ninfer::cli::Options configured =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--thinking-budget", "37"});
    failures += check(configured.thinking_budget == 37,
                      "--thinking-budget did not preserve its positive value");
    failures +=
        check(ninfer::cli::usage_text("ninfer-cli").find("--thinking-budget") != std::string::npos,
              "CLI help omits --thinking-budget");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--thinking-budget", "0"});
                      }),
                      "zero --thinking-budget was accepted");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--thinking-budget", "8", "--no-thinking"});
                      }),
                      "--thinking-budget was accepted with --no-thinking");
    const ninfer::cli::Options with_effort =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--thinking-budget", "8",
               "--reasoning-effort", "medium"});
    failures += check(with_effort.thinking_budget == 8 && with_effort.reasoning_effort,
                      "thinking budget did not coexist with reasoning effort");
    const ninfer::cli::Options dflash_vision =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--vision", "--spec", "dflash",
               "--draft-tokens", "7"});
    failures += check(dflash_vision.enable_vision &&
                          dflash_vision.speculative.backend == ninfer::SpeculativeBackend::DFlash &&
                          dflash_vision.speculative.draft_tokens == 7,
                      "CLI did not preserve the combined DFlash and Vision startup features");
    failures += check(parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--vision"})
                              .vision_max_item_tokens == ninfer::kMaximumVisionItemTokens,
                      "CLI did not default to the native Vision item extent");
    failures += check(parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--vision",
                             "--vision-max-item-tokens", "4096"})
                              .vision_max_item_tokens == 4096,
                      "CLI lost --vision-max-item-tokens");
    for (const char* bad : {"0", "255", "16385", "x"}) {
        failures += check(rejects([&] {
                              (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                           "--vision", "--vision-max-item-tokens", bad});
                          }),
                          "CLI accepted an invalid --vision-max-item-tokens");
    }
    for (const auto k : {1U, 2U, 7U, 15U}) {
        const auto dflash2 = parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--spec",
                                    "dflash2", "--draft-tokens", std::to_string(k)});
        failures += check(dflash2.speculative.backend == ninfer::SpeculativeBackend::DFlash2 &&
                              dflash2.speculative.draft_tokens == k,
                          "CLI did not preserve the DFlash2 draft count");
    }
    for (const auto k : {0U, 16U}) {
        failures +=
            check(rejects([&] {
                      (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--spec",
                                   "dflash2", "--draft-tokens", std::to_string(k)});
                  }),
                  "CLI accepted an unsupported DFlash2 draft count");
    }
    const ninfer::cli::Options nvfp4 =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--kv-dtype", "nvfp4"});
    failures += check(nvfp4.kv_cache == ninfer::KvCacheStorage::Nvfp4Group16,
                      "--kv-dtype nvfp4 did not select group-16 NVFP4 KV");
    const ninfer::cli::Options k8v4 =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--kv-dtype", "k8v4"});
    failures += check(k8v4.kv_cache == ninfer::KvCacheStorage::Fp8KeyNvfp4Value,
                      "--kv-dtype k8v4 did not select asymmetric K8V4 KV");
    const std::string help = ninfer::cli::usage_text("ninfer-cli");
    failures +=
        check(help.find("nvfp4") != std::string::npos && help.find("k8v4") != std::string::npos,
              "CLI help omits a production KV storage mode");
    const ninfer::cli::Options logging =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--log-level", "debug"});
    failures += check(logging.log_level == ninfer::product::LogLevel::Debug,
                      "CLI log level was not parsed");
    failures += check(help.find("--log-level") != std::string::npos,
                      "CLI help omits the log-level control");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--log-level", "verbose"});
                      }),
                      "CLI accepted an unknown log level");
    failures +=
        check(rejects([] {
                  (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--top-k", "21"});
              }),
              "CLI accepted top_k beyond the executable candidate domain");
    return failures == 0 ? 0 : 1;
}
