if (Sys.getenv("HOME") != "/home/jovyan" &
  file.exists(paste0(getwd(), "/.vscode"))) {
  Sys.setenv(TERM_PROGRAM = "vscode")
  Sys.setenv(DOWNLOAD_STATIC_LIBV8 = 1)
  source(file.path(
    Sys.getenv(if
    (.Platform$OS.type == "windows") {
      "USERPROFILE"
    } else {
      "HOME"
    }),
    ".vscode-R", "init.R"
  ))
}

if (interactive() && Sys.getenv("TERM_PROGRAM") == "vscode") {
  options(vsc.browser = "Beside")
  options(vsc.viewer = "Beside")
  options(vsc.page_viewer = "Beside")
  options(vsc.view = "Beside")
  if ("httpgd" %in% .packages(all.available = TRUE)) {
    options(vsc.plot = FALSE)
    options(html_type = "html")
    options(vsc.helpPanel = FALSE)
    options(device = function(...) {
      # On Tsa the fonts are not automatically recognized by hgd
      httpgd::hgd()
      httpgd::hgd_svg()
      .vsc.browser(httpgd::hgd_url(), viewer = "Beside")
    })
  } else {
    options(vsc.plot = "Beside")
    options(vsc.dev.args = list(width = 800, height = 600))
  }
}

options(error = function() {
  calls <- sys.calls()
  if (length(calls) >= 2L) {
    sink(stderr())
    on.exit(sink(NULL))
    cat("Backtrace:\n")
    calls <- rev(calls[-length(calls)])
    for (i in seq_along(calls)) {
      cat(i, ": ", deparse(calls[[i]], nlines = 1L), "\n", sep = "")
    }
  }
  if (!interactive()) {
    q(status = 1)
  }
})

rs <- function(args = NULL) {
  getOption("rchitect.py_tools")$attach()
  os <- import("os")
  sys <- import("sys")
  os$execv(sys$executable, c("sys$executable", "-m", "radian", args))
}
