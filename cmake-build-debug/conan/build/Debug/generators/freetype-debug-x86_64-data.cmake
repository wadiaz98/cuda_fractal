########### AGGREGATED COMPONENTS AND DEPENDENCIES FOR THE MULTI CONFIG #####################
#############################################################################################

set(freetype_COMPONENT_NAMES "")
if(DEFINED freetype_FIND_DEPENDENCY_NAMES)
  list(APPEND freetype_FIND_DEPENDENCY_NAMES PNG ZLIB BZip2 brotli)
  list(REMOVE_DUPLICATES freetype_FIND_DEPENDENCY_NAMES)
else()
  set(freetype_FIND_DEPENDENCY_NAMES PNG ZLIB BZip2 brotli)
endif()
set(PNG_FIND_MODE "NO_MODULE")
set(ZLIB_FIND_MODE "NO_MODULE")
set(BZip2_FIND_MODE "NO_MODULE")
set(brotli_FIND_MODE "NO_MODULE")

########### VARIABLES #######################################################################
#############################################################################################
set(freetype_PACKAGE_FOLDER_DEBUG "C:/Users/fing.labcom/.conan2/p/b/freet129b06f9fe513/p")
set(freetype_BUILD_MODULES_PATHS_DEBUG "${freetype_PACKAGE_FOLDER_DEBUG}/lib/cmake/conan-official-freetype-variables.cmake")


set(freetype_INCLUDE_DIRS_DEBUG )
set(freetype_RES_DIRS_DEBUG )
set(freetype_DEFINITIONS_DEBUG )
set(freetype_SHARED_LINK_FLAGS_DEBUG )
set(freetype_EXE_LINK_FLAGS_DEBUG )
set(freetype_OBJECTS_DEBUG )
set(freetype_COMPILE_DEFINITIONS_DEBUG )
set(freetype_COMPILE_OPTIONS_C_DEBUG )
set(freetype_COMPILE_OPTIONS_CXX_DEBUG )
set(freetype_LIB_DIRS_DEBUG "${freetype_PACKAGE_FOLDER_DEBUG}/lib")
set(freetype_BIN_DIRS_DEBUG )
set(freetype_LIBRARY_TYPE_DEBUG STATIC)
set(freetype_IS_HOST_WINDOWS_DEBUG 1)
set(freetype_LIBS_DEBUG freetype)
set(freetype_SYSTEM_LIBS_DEBUG )
set(freetype_FRAMEWORK_DIRS_DEBUG )
set(freetype_FRAMEWORKS_DEBUG )
set(freetype_BUILD_DIRS_DEBUG )
set(freetype_NO_SONAME_MODE_DEBUG FALSE)


# COMPOUND VARIABLES
set(freetype_COMPILE_OPTIONS_DEBUG
    "$<$<COMPILE_LANGUAGE:CXX>:${freetype_COMPILE_OPTIONS_CXX_DEBUG}>"
    "$<$<COMPILE_LANGUAGE:C>:${freetype_COMPILE_OPTIONS_C_DEBUG}>")
set(freetype_LINKER_FLAGS_DEBUG
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${freetype_SHARED_LINK_FLAGS_DEBUG}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${freetype_SHARED_LINK_FLAGS_DEBUG}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${freetype_EXE_LINK_FLAGS_DEBUG}>")


set(freetype_COMPONENTS_DEBUG )